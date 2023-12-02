#include "Viewport.h"
#include <GL/gl3w.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


Viewport::Viewport(const Shader &sh) : size({ -1,-1 }), shader(sh), camera(), grid(), gridShader(), axisShader() {
	//model = new Model("assets/Survival_BackPack_2/backpack.obj");
	model = new Model("assets/pony-cartoon/source/Pony_cartoon.obj");
	camera.setPosition(glm::vec3(0, 2, 6));
	camera.pitch = -10.0f;
	camera.updateView();
	glEnable(GL_DEPTH_TEST); 
	glEnable(GL_MULTISAMPLE);
	gridShader.setVertex("shader/forward_vertex.glsl");
	gridShader.setGeometry("shader/grid_geometry.glsl");
	gridShader.setFragment("shader/color_fragment.glsl");
	gridShader.link();
	gridShader.use();
	gridShader.setVec3("color", glm::vec3(0.3f));


	axisShader.setVertex("shader/forward_vertex.glsl");
	axisShader.setGeometry("shader/axis_geometry.glsl");
	axisShader.setFragment("shader/in_color_fragment.glsl");
	axisShader.link();
	axisShader.use();
	axisShader.setFloat("size", 0.2f);
}

void Viewport::setShader(const Shader& sh) {
	shader = sh;
}

void Viewport::updateCameraProjection() {
	camera.setPerspective(fov, size.x / size.y, 0.1f, 1000.0f);
}

void Viewport::updateFramebuffer() {
	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();
	ImVec2 newViewportSize = ImVec2(vMax.x - vMin.x, vMax.y - vMin.y);
	if (size.x == newViewportSize.x && size.y == newViewportSize.y)
		return;
	size = newViewportSize;
	updateCameraProjection();

	glDeleteFramebuffers(1, &framebuffer);
	glDeleteFramebuffers(1, &intermediateFBO);
	glDeleteTextures(1, &textureColorbuffer);
	glDeleteTextures(1, &screenTexture);
	glDeleteRenderbuffers(1, &rbo);

	//Generate a framebuffer to render to
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	glGenTextures(1, &textureColorbuffer);
	int samples = 4;
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorbuffer);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGB, size.x, size.y, GL_TRUE);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

	// attach it to currently bound framebuffer object
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorbuffer, 0);


	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, size.x, size.y);
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screenTexture, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Intermediate framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Viewport::render(float deltaTime) {
	ImGui::Begin("Viewport");
	updateFramebuffer();

	//update camera
	if (ImGui::IsWindowFocused()) {
		if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
			ImVec2 mousePos = ImGui::GetMousePos();
			if (firstMouse) {
				firstMouse = false;
				lastMousePos = mousePos;
			}
			else {
				float sensitivity = 0.1f;
				ImVec2 offset = { sensitivity * (mousePos.x - lastMousePos.x), sensitivity * (lastMousePos.y - mousePos.y) };
				lastMousePos = mousePos;

				camera.yaw += offset.x;
				camera.pitch += offset.y;

				if (camera.pitch > 89.0f)
					camera.pitch = 89.0f;
				if (camera.pitch < -89.0f)
					camera.pitch = -89.0f;

				camera.updateView();
			}
		}
		else
			firstMouse = true;

		float cameraSpeed = 10.0f;
		cameraSpeed *= deltaTime;
		//std::cout << deltaTime << std::endl;
		glm::vec3 moveDir = glm::vec3(0);
		if (ImGui::IsKeyDown(ImGuiKey_W))
			moveDir += camera.direction;
		if (ImGui::IsKeyDown(ImGuiKey_A))
			moveDir += camera.right;
		if (ImGui::IsKeyDown(ImGuiKey_S))
			moveDir -= camera.direction;
		if (ImGui::IsKeyDown(ImGuiKey_D))
			moveDir -= camera.right;

		if (moveDir != glm::vec3(0)) {
			moveDir = glm::normalize(moveDir) * cameraSpeed;
			camera.position += moveDir;
			camera.updateView();
		}
	}
	else
		firstMouse = true;

	//Setup Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(0, 0, size.x, size.y);
	glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Draw 
	if (settings.wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	


	shader.use();
	camera.useInShader(shader);
	//glm::vec3 lightPos = glm::vec3(5 * glm::sin(ImGui::GetTime()), 0, 5 * glm::cos(ImGui::GetTime()));

	//shader.setVec3("pLight.position", lightPos); 
	//shader.setVec3("pLight.ambient", glm::vec3(0.0f));
	//shader.setVec3("pLight.diffuse", glm::vec3(0.6f)); // darken diffuse light a bit
	//shader.setVec3("pLight.specular", glm::vec3(1.0f));

	//dirLight = glm::vec3(glm::sin(ImGui::GetTime()), glm::cos(ImGui::GetTime()), 0);
	
	shader.setVec3("dLight.direction", dirLight);
	shader.setVec3("dLight.ambient", clearColor);
	shader.setVec3("dLight.diffuse", glm::vec3(1.0f)); // darken diffuse light a bit
	shader.setVec3("dLight.specular", glm::vec3(1.0f));
	shader.setVec3("viewPos", camera.position);	
	
	shader.setFloat("material.shininess", 32);
	shader.setMat4("transform", glm::mat4(1));
	model->render(shader);

	//render grid and axies
	gridShader.use();
	camera.useInShader(gridShader);
	glDrawArrays(GL_POINTS, 0, 1);

	glLineWidth(3.0f);
	axisShader.use();
	camera.useInShader(axisShader);
	glDrawArrays(GL_POINTS, 0, 1);
	glLineWidth(1.0f);



	glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediateFBO);
	glBlitFramebuffer(0, 0, size.x, size.y, 0, 0, size.x, size.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	//unbind Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	ImGui::Image((void*)screenTexture, size, { 0, 1 }, { 1, 0 });
	ImGui::End();
}

Viewport::~Viewport()
{
	delete model;
	glDeleteFramebuffers(1, &framebuffer);
	glDeleteFramebuffers(1, &intermediateFBO);
	glDeleteTextures(1, &textureColorbuffer);
	glDeleteTextures(1, &screenTexture);
	glDeleteRenderbuffers(1, &rbo);
	std::cout << "Viewport unloaded" << std::endl;
}
