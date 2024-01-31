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
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 64));
	cudaDeviceGetLimit(&currStackSize, cudaLimitStackSize);
	std::cout << "new StackSize:" << currStackSize << std::endl;

	size_t currMallocSize;
	cudaDeviceGetLimit(&currMallocSize, cudaLimitMallocHeapSize);
	std::cout << "currMallocSize:" << currMallocSize << std::endl;
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1llu << 29));
	cudaDeviceGetLimit(&currMallocSize, cudaLimitMallocHeapSize);
	std::cout << "new MallocSize:" << currMallocSize << std::endl;

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << "total:" << total << ", free:" << free << std::endl;

	viewport = new DynamicViewport();
	//glfwSwapInterval(0);
}

void App::showTextureStack(MultiMaterial* mat, const char* name, TextureType type) {
	if (ImGui::TreeNode(name))
	{
		if (mat->textures[type] == -1)
			ImGui::InputFloat3("Base Color", &(mat->colors[type].x));
		else
		{
			ImGui::Image((void*)mat->textures[type], ImVec2(200, 200), { 0, 1 }, { 1, 0 });
		}
		ImGui::TreePop();
	}
}

void App::renderUI(ImGuiIO& io) {
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");
	ImGui::Text("Framerate %.3f ms/frame (%.1f FPS)", 1000.0f * io.DeltaTime, 1.0f / io.DeltaTime);
	if (ImGui::InputInt("Renderer", (int*)&(viewport->settings.renderer))) {
		if (viewport->settings.renderer > renderTypeCount - 1)
			viewport->settings.renderer = renderTypeCount - 1;
		if (viewport->settings.renderer < 0)
			viewport->settings.renderer = 0;
		viewport->updateSettings();
	}

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
	ImGui::Text("Rasterizer");

	if (ImGui::TreeNode("Rasterizer")) {
		Rasterizer* rasterizer = (Rasterizer*)(viewport->camera->getRenderer(renderTypeRasterize));
		ImGui::Checkbox("Draw Normals", &(rasterizer->showNormals));
		ImGui::InputFloat("Normals Length", &(rasterizer->normalsLength));
		ImGui::Checkbox("show AABBs", &(rasterizer->showAABBs));
		ImGui::Checkbox("Wireframe", &(rasterizer->wireframe));
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Raytracer")) {
		RayTracer* rayTracer = (RayTracer*)(viewport->camera->getRenderer(renderTypeRayTrace));
		if (ImGui::InputInt("Samples", &rayTracer->samples)) rayTracer->updateView();
		if (ImGui::InputInt("Max Steps", &rayTracer->max_steps)) rayTracer->updateView();
		ImGui::TreePop();
	}


	if (ImGui::TreeNode("Materials"))
	{
		for (int i = 0; i < viewport->scene.materials.size(); i++)
		{
			// Use SetNextItemOpen() so set the default state of a node to be open. We could
			// also use TreeNodeEx() with the ImGuiTreeNodeFlags_DefaultOpen flag to achieve the same thing!
			if (i == 0)
				ImGui::SetNextItemOpen(true, ImGuiCond_Once);

			MultiMaterial* mat = viewport->scene.materials[i];
			if (ImGui::TreeNode((void*)(intptr_t)i, mat->name, i))
			{
				ImGui::Text("Opacity %.2f", mat->opacity);
				ImGui::Text("Refraction Index %.2f", mat->refractionIndex);
				ImGui::InputFloat("Shininess", &mat->shininess);
				ImGui::Text("Shininess Strength %.2f", mat->shininessStrength);

				showTextureStack(mat, "Diffuse", textureTypeDiffuse);
				showTextureStack(mat, "Specular", textureTypeSpecular);
				showTextureStack(mat, "Normal", textureTypeNormal);
				ImGui::TreePop();
			}
		}
		ImGui::TreePop();
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
