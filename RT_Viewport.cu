#include "RT_Viewport.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Sphere.h"

#include "raytracing.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


RT_Viewport::RT_Viewport() : size({ -1,-1 }) {
	checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(RT_Camera)));

	checkCudaErrors(cudaMalloc((void**)&objects, 2*sizeof(Hitable*)));
	checkCudaErrors(cudaMalloc((void**)&scene, sizeof(Hitable*)));
	init_scene<<<1, 1 >>>(scene, objects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

//returns wether to the current framebuffer is drawable
bool RT_Viewport::updateFramebuffer() {
	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();
	ImVec2 newViewportSize = ImVec2(vMax.x - vMin.x, vMax.y - vMin.y);
	if (size.x == newViewportSize.x && size.y == newViewportSize.y)
		return resizeFinished;

	//is currently resizing? -> not drawable
	if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
		return false;

	resizeFinished = false;
	size = newViewportSize;

	//is window hidden? -> not drawable
	if (size.x == 0 || size.y == 0)
		return false;


	cam->update(size.x, size.y);
	std::cout << "new texture: " << size.x << " " << size.y << std::endl;

	glDeleteTextures(1, &texture);
	glGenTextures(1, &texture);
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &texture);

	glBindTexture(GL_TEXTURE_2D, texture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	auto e = cudaGraphicsGLRegisterImage(&gfxRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	checkCudaErrors(e);

	//init randoms
	if(d_rand_state != nullptr)
		checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, size.x * size.y * sizeof(curandState)));

	dim3 blocks(size.x / blockW + 1, size.y / blockH + 1);
	dim3 threads(blockW, blockH);
	renderInit << <blocks, threads >> > (size.x, size.y, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	accumulation = 0;


	resizeFinished = true;
	return true;
}

void RT_Viewport::render(float deltaTime) {
	ImGui::Begin("RT Viewport");
	bool fbDrawable = updateFramebuffer();
	if (size.x <= 0 || size.y <= 0) {
		ImGui::End();
		return;
	}
	if (fbDrawable)
		invokeRenderProcedure();

	ImGui::Image((void*)texture, size); // , { 0, 1 }, { 1, 0 });
	ImGui::End();
}


void RT_Viewport::invokeRenderProcedure() {
	
	// Render our buffer
	dim3 blocks(size.x / blockW + 1, size.y / blockH + 1);
	dim3 threads(blockW, blockH);

	auto e = cudaGraphicsMapResources(1, &gfxRes);
	checkCudaErrors(e);

	cudaArray_t viewCudaArray;
	e = cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, gfxRes, 0, 0);
	checkCudaErrors(e);
	cudaResourceDesc viewCudaArrayResourceDesc;
	{
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
	}
	cudaSurfaceObject_t viewCudaSurfaceObject;
	e = cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
	checkCudaErrors(e);
	accumulation++;
	renderImage <<<blocks, threads>>> (viewCudaSurfaceObject, size.x, size.y, cam, scene, d_rand_state, samples, max_steps, accumulation);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	e = cudaDestroySurfaceObject(viewCudaSurfaceObject);
	checkCudaErrors(e);

	e = cudaGraphicsUnmapResources(1, &gfxRes);
	checkCudaErrors(e);

	e = cudaStreamSynchronize(0);
	checkCudaErrors(e);
}

RT_Viewport::~RT_Viewport()
{
	checkCudaErrors(cudaDeviceSynchronize());
	free_scene<<<1, 1 >>>(scene, objects, 2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(objects));
	checkCudaErrors(cudaFree(scene));
	checkCudaErrors(cudaFree(cam));
	glDeleteTextures(1, &texture);
}
