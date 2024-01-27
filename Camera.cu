#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>
#include "raytracing.h"

Camera::Camera(Scene& scene) {
	renderer[renderTypeRasterize] = new Rasterizer(scene);
	renderer[renderTypeRayTrace] = new RayTracer(scene);
}

Camera::~Camera() {
	for (int i = 0; i < renderTypeCount; i++)
		delete renderer[i];
}

void Camera::setPosition(glm::vec3 pos) {
	position = pos;
	update();
}

void Camera::update() {
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction = glm::normalize(direction);

	position = focalPoint - distance * direction;

	glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::normalize(glm::cross(worldUp, direction));
	up = glm::cross(direction, right);

	for (int i = 0; i < renderTypeCount; i++)
		renderer[i]->setViewVectors(position, direction, right, up);
}
void Camera::resize(const int& width, const int& height) {
	for (int i = 0; i < renderTypeCount; i++)
		renderer[i]->resize(width, height, fov, nearPlane, farPlane);
}
uint Camera::render() {
	return renderer[activeRenderer]->render();
}

void Rasterizer::updateView() {
	view = glm::lookAt(position, position + direction, up);
}

void RayTracer::updateView() {
	// Calculate the location of the upper left pixel.
	Vec3 upperLeft = position + (direction * nearPlane) - (right * viewportU / 2) - (up * viewportV / 2);
	pixel00_loc = upperLeft + 0.5 * (pixelDeltaU * right + pixelDeltaV * up);

	accumulation = 0;
	if (deviceCopy == nullptr)
		cudaMallocManaged((void**)&deviceCopy, sizeof(RayTracer));
	*deviceCopy = *this;
}

void Rasterizer::resize(const int& width, const int& height, const float& fov, const float& nearPlane, const float& farPlane) {
	imageWidth = width;
	imageHeight = height;
	float aspect = (float)width / (float)height;
	projection = glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);

	glDeleteFramebuffers(1, &framebuffer);
	glDeleteFramebuffers(1, &intermediateFBO);
	glDeleteTextures(1, &textureColorbuffer);
	glDeleteTextures(1, &screenTexture);
	glDeleteRenderbuffers(1, &rbo);

	//Generate a framebuffer to render to
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	//gen multisample texture (anti-aliasing)
	glGenTextures(1, &textureColorbuffer);
	int samples = 4;
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorbuffer);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGB, width, height, GL_TRUE);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

	// attach it to currently bound framebuffer object
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorbuffer, 0);

	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// configure second post-processing framebuffer
	glGenFramebuffers(1, &intermediateFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, intermediateFBO);
	// create a color attachment texture
	glGenTextures(1, &screenTexture);
	glBindTexture(GL_TEXTURE_2D, screenTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screenTexture, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Intermediate framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RayTracer::resize(const int& width, const int& height, const float& fov, const float& nearPlane, const float& farPlane) {
	this->fov = fov;
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;
	imageWidth = width;
	imageHeight = height;
	float aspect = (float)width / (float)height;
	viewportV = 2 * nearPlane * glm::tan(glm::radians(fov) / 2.0f);
	viewportU = -viewportV * aspect;
	pixelDeltaU = viewportU / width;
	pixelDeltaV = viewportV / height;

	//update image dimensions
	glDeleteTextures(1, &image);
	glGenTextures(1, &image);
	glBindTexture(GL_TEXTURE_2D, image);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	renderImage.init(image);

	//init randoms
	if (randomStates != nullptr)
		checkCudaErrors(cudaFree(randomStates));
	checkCudaErrors(cudaMalloc((void**)&randomStates, width * height * sizeof(curandState)));

	dim3 blocks(width / blockW + 1, height / blockH + 1);
	dim3 threads(blockW, blockH);
	render_init << <blocks, threads >> > (width, height, randomStates);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	accumulation = 0;

	updateView();
}



uint RayTracer::render() {
	dim3 blocks(imageWidth / blockW + 1, imageHeight / blockH + 1);
	dim3 threads(blockW, blockH);

	accumulation++;
	environment.map();
	renderImage.map();

	render_image << <blocks, threads >> > (renderImage.surfaceObject, imageWidth, imageHeight, deviceCopy, scene, environment.texObject, randomStates, samples, max_steps, accumulation);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	environment.unmap();
	renderImage.unmap();
	return image;
}

uint Rasterizer::render() {
	//Setup Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(0, 0, imageWidth, imageHeight);
	glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Draw 
	bool wireframe = false;
	if (wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	shader.use();
	shader.setMat4("view", view);
	shader.setMat4("projection", projection);

	glm::vec3 dirLight = glm::vec3(1, -1, -1);
	shader.setVec3("dLight.direction", dirLight);
	shader.setVec3("dLight.ambient", clearColor);
	shader.setVec3("dLight.diffuse", glm::vec3(1.0f)); // darken diffuse light a bit
	shader.setVec3("dLight.specular", glm::vec3(1.0f));
	shader.setVec3("viewPos", position);
	shader.setFloat("material.shininess", 32);

	scene.render(shader, false);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediateFBO);
	glBlitFramebuffer(0, 0, imageWidth, imageHeight, 0, 0, imageWidth, imageHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	//unbind Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return screenTexture;
}