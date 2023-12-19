#include "Application.h"
#include "imgui.h"
#include <format>
#include <iostream>
#include <GL/gl3w.h>
#include "Viewport.h"
#include "imgui.h"
#include <GLFW/glfw3.h>

App::App() : shader("shader/vertex.glsl", "shader/fragment.glsl") {
	//viewport = new Viewport(shader);
	rt_viewport = new RT_Viewport();
	glfwSwapInterval(0);
}
	

void App::renderUI(ImGuiIO& io) {
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
	
	/*ImGui::Checkbox("Wireframe", &(viewport->settings.wireframe));
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
	ImGui::Checkbox("Draw AABBs", &(viewport->settings.drawAABBs));
	ImGui::Checkbox("Draw Normals", &(viewport->settings.drawNormals));
	ImGui::InputFloat("Normals Length", &(viewport->settings.viewNormalsLength));
	ImGui::InputFloat3("DirLight", glm::value_ptr(viewport->dirLight));*/

	ImGui::InputInt("Samples", &(rt_viewport->samples), 1);
	ImGui::InputInt("Max Steps", &(rt_viewport->max_steps), 1);

	ImGui::End();

	//viewport->render(io.DeltaTime);
	rt_viewport->render(io.DeltaTime);
}

App::~App()
{
	//delete rt_viewport;
	//delete viewport;
	shader.destroy();
}
