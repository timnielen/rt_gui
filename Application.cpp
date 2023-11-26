#include "Application.h"
#include "imgui.h"
#include <format>
#include <iostream>
#include <GL/gl3w.h>
#include "Viewport.h"
#include "imgui.h"

App::App() : shader("shader/vertex.glsl", "shader/fragment.glsl") {
	viewport = new Viewport(shader);
}
	

void App::renderUI(ImGuiIO& io) {
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
	ImGui::Checkbox("Wireframe", &(viewport->settings.wireframe));
	if (ImGui::Button("Reload shader")) {
		shader.destroy();
		shader = Shader("shader/vertex.glsl", "shader/fragment.glsl");
		viewport->setShader(shader);
	}
	if (ImGui::SliderFloat("FOV", &(viewport->fov), 1, 180, "%.0f")) {
		viewport->updateCameraProjection();
	}
	if (ImGui::InputFloat3("Camera position", glm::value_ptr(viewport->camera.position))) {
		viewport->camera.updateView();
	}
	ImGui::End();

	viewport->render();
}

App::~App()
{
	delete viewport;
	shader.destroy();
}
