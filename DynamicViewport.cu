#include "DynamicViewport.h"
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


DynamicViewport::DynamicViewport() : size({ -1,-1 }) {
	//scene = Scene("assets/ship/source/full_scene.fbx");
	scene = Scene("assets/Survival_BackPack_2/backpack.obj", false);
	//scene = Scene("assets/test.obj");
	camera = new Camera(scene);
	camera->setPosition(glm::vec3(0, 0, camera->distance));
}

DynamicViewport::~DynamicViewport() {
	delete camera;
}

void DynamicViewport::updateFramebuffer() {
	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();
	ImVec2 newViewportSize = ImVec2(vMax.x - vMin.x, vMax.y - vMin.y);
	bool resized = size.x != newViewportSize.x || size.y != newViewportSize.y;
	if (!resized)
		return;
	size = newViewportSize;

	if (size.x == 0 || size.y == 0)
		return;

	camera->resize(size.x, size.y);
}

void DynamicViewport::handleUserInput(float deltaTime) {
	if (ImGui::IsWindowFocused()) {
		if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
			ImVec2 mousePos = ImGui::GetMousePos();
			if (firstMouse) {
				firstMouse = false;
				lastMousePos = mousePos;
			}
			else {
				ImVec2 offset = { mouseSensitivity * (mousePos.x - lastMousePos.x), mouseSensitivity * (lastMousePos.y - mousePos.y) };
				lastMousePos = mousePos;

				camera->yaw += offset.x;
				camera->pitch += offset.y;

				if (camera->pitch > 89.0f)
					camera->pitch = 89.0f;
				if (camera->pitch < -89.0f)
					camera->pitch = -89.0f;

				camera->update();
			}
		}
		else
			firstMouse = true;

		float cameraSpeed = 2.0f;
		cameraSpeed *= deltaTime;
		//std::cout << deltaTime << std::endl;
		glm::vec3 moveDir = glm::vec3(0);
		if (ImGui::IsKeyDown(ImGuiKey_W))
			moveDir += camera->direction;
		if (ImGui::IsKeyDown(ImGuiKey_A))
			moveDir += camera->right;
		if (ImGui::IsKeyDown(ImGuiKey_S))
			moveDir -= camera->direction;
		if (ImGui::IsKeyDown(ImGuiKey_D))
			moveDir -= camera->right;

		if (moveDir != glm::vec3(0)) {
			moveDir = glm::normalize(moveDir) * cameraSpeed;
			camera->focalPoint += moveDir;
			camera->update();
		}

		int mouseWheel = ImGui::GetIO().MouseWheel;
		if (mouseWheel != 0) {
			camera->distance -= mouseWheel;
			camera->update();
		};
	}
	else
		firstMouse = true;
}

void DynamicViewport::draw(float deltaTime) {
	ImGui::Begin("Viewport");
	updateFramebuffer();

	handleUserInput(deltaTime);

	if (size.x > 1 && size.y > 1)
	{
		unsigned int image = camera->render();
		ImGui::Image((void*)image, size, { 0, 1 }, { 1, 0 });
	}
	ImGui::End();
}
