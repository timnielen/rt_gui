#include "Viewport.h"
#include <GL/gl3w.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

glm::vec3 cubePositions[] = {
	glm::vec3(2.0f, 5.0f, -15.0f),
	glm::vec3(-1.5f, -2.2f, -2.5f),
	glm::vec3(-3.8f, -2.0f, -12.3f),
	glm::vec3(2.4f, -0.4f, -3.5f),
	glm::vec3(-1.7f, 3.0f, -7.5f),
	glm::vec3(1.3f, -2.0f, -2.5f),
	glm::vec3(1.5f, 2.0f, -2.5f),
	glm::vec3(1.5f, 0.2f, -1.5f),
	glm::vec3(-1.3f, 1.0f, -1.5f)
};

Viewport::Viewport(const Shader &sh) : size({ -1,-1 }), shader(sh), camera(), object() {
	camera.setPosition(glm::vec3(0, 0, 3));
	glEnable(GL_DEPTH_TEST);
}

void Viewport::setShader(const Shader& sh) {
	shader = sh;
}

void Viewport::updateCameraProjection() {
	camera.setPerspective(fov, size.x / size.y, 0.1f, 100.0f);
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
	glDeleteTextures(1, &textureColorbuffer);
	glDeleteRenderbuffers(1, &rbo);

	//Generate a framebuffer to render to
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// attach it to currently bound framebuffer object
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, size.x, size.y);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Viewport::render() {
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

		float cameraSpeed = 0.1f;
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
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Draw 
	if (settings.wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	


	shader.use();
	camera.useInShader(shader);
	shader.setVec3("lightPos", glm::vec3(0));
	shader.setVec3("lightColor", glm::vec3(1));
	shader.setVec3("objectColor", glm::vec3(1,0.5f,0.1f));
	

	for (unsigned int i = 0; i < 9; i++) {
		float angle = 20.0f * i;
		object.setScale(1);
		object.setRotationAngle(glm::radians(angle));
		object.setRotationAxis(glm::vec3(1.0f, 0.3f, 0.5f));
		object.setPosition(cubePositions[i]);
		//object.setScale(glm::vec3(1, 1, 5));
		object.render(shader);
	}

	object.setPosition(glm::vec3(0));
	object.setScale(0.1f);
	shader.setVec3("objectColor", glm::vec3(100));
	object.render(shader);


	//unbind Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	ImGui::Image((void*)textureColorbuffer, size, { 0, 1 }, { 1, 0 });
	ImGui::End();
}

Viewport::~Viewport()
{
	glDeleteFramebuffers(1, &framebuffer);
	glDeleteTextures(1, &textureColorbuffer);
	glDeleteRenderbuffers(1, &rbo);
	std::cout << "Viewport unloaded" << std::endl;
}
