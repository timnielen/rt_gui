#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>
#include "raytracing.h"

void Camera::setPosition(glm::vec3 pos) {
	position = pos;
	updateView();
}

void Rasterizer::resize(int width, int height) {
	float aspect = (float)width / (float)height;
	projection = glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
}

void RayTracer::resize(int width, int height) {
	imageWidth = width;
	imageHeight = height;
	float aspect = (float)width / (float)height;
	viewportU = -2 * nearPlane * glm::tan(glm::radians(fov) / 2.0f);
	viewportV = -viewportU / aspect;
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

void Rasterizer::updateView() {
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction = glm::normalize(direction);

	glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::normalize(glm::cross(worldUp, direction));
	up = glm::cross(direction, right);

	view = glm::lookAt(position, position + direction, up);
}

void RayTracer::updateView() {
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction = glm::normalize(direction);

	glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::normalize(glm::cross(worldUp, direction));
	up = glm::cross(direction, right);

	// Calculate the location of the upper left pixel.
	Vec3 upperLeft = position + (direction * nearPlane) - (right * viewportU / 2) - (up * viewportV / 2);
	pixel00_loc = upperLeft + 0.5 * (pixelDeltaU * right + pixelDeltaV * up);

	accumulation = 0;
	if (deviceCopy == nullptr)
		cudaMallocManaged((void**)&deviceCopy, sizeof(RayTracer));
	*deviceCopy = *this;
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
	return 0;
}