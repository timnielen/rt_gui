#include <GL/gl3w.h>
#include "Application.h"
#include <GLFW/glfw3.h>

#include <format>
#include <iostream>
#include "cuda_helper.h"


App::App() {
	unsigned int dCount = 0;
	int devices[10];
	cudaDeviceProp prop;
	checkCudaErrors(cudaGLGetDevices(&dCount, devices, 10, cudaGLDeviceListAll));
	std::cout << "Devices used to render image (OpenGL):" << std::endl;
	for (int i = 0; i < dCount; i++) {
		cudaGetDeviceProperties(&prop, devices[i]);
		std::cout << devices[i] << "\t" << prop.name << std::endl;
	}
	int cudaDevice;
	cudaGetDevice(&cudaDevice);
	cudaGetDeviceProperties(&prop, cudaDevice);
	std::cout << "Cuda device:" << std::endl;
	std::cout << cudaDevice << "\t" << prop.name << std::endl;
	std::cout << "totalGlobalMem: " << "\t" << prop.totalGlobalMem << std::endl;
	std::cout << "sharedMemPerBlock: " << "\t" << prop.sharedMemPerBlock << std::endl;
	std::cout << "warpSize: " << "\t" << prop.warpSize << std::endl;

	size_t currStackSize;
	cudaDeviceGetLimit(&currStackSize, cudaLimitStackSize);
	std::cout << "currStackSize:" << currStackSize << std::endl;
	cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 8);

	size_t currMallocSize;
	cudaDeviceGetLimit(&currMallocSize, cudaLimitMallocHeapSize);
	std::cout << "currMallocSize:" << currMallocSize << std::endl;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2 * currMallocSize);

	viewport = new DynamicViewport();
	//glfwSwapInterval(0);
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
	ImGui::InputFloat3("DirLight", glm::value_ptr(viewport->dirLight));

	ImGui::InputInt("Samples", &(rt_viewport->samples), 1);
	ImGui::InputInt("Max Steps", &(rt_viewport->max_steps), 1);*/

	if (ImGui::InputInt("Renderer", (int*)&(viewport->settings.renderer))) {
		if (viewport->settings.renderer > CAMERA_RENDERER_COUNT - 1)
			viewport->settings.renderer = CAMERA_RENDERER_COUNT - 1;
		if (viewport->settings.renderer < 0)
			viewport->settings.renderer = 0;
		viewport->updateSettings();
	}
	ImGui::End();

	viewport->draw(io.DeltaTime);
	//rt_viewport->render(io.DeltaTime);
}

App::~App()
{
	//delete rt_viewport;
	delete viewport;
}
