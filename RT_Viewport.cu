#include "RT_Viewport.h"
#include <iostream>



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
}

void RT_Viewport::updateFramebuffer() {
	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();
	ImVec2 newViewportSize = ImVec2(vMax.x - vMin.x, vMax.y - vMin.y);
	if (size.x == newViewportSize.x && size.y == newViewportSize.y)
		return;
	size = newViewportSize;
	if (size.x == 0 || size.y == 0)
		return;

	int floatCount = size.x * size.y * 3;
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
}
__global__ void renderImage(cudaSurfaceObject_t s, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;
	float4 data = make_float4(float(i) / max_x, float(j) / max_y, 0.2f, 1.0f);
	surf2Dwrite(data, s, i * sizeof(float4), j);

}



void RT_Viewport::render(float deltaTime) {
	ImGui::Begin("RT Viewport");
	updateFramebuffer();
	if (size.x <= 0 || size.y <= 0) {
		ImGui::End();
		return;
	}
	invokeRenderProcedure();

	ImGui::Image((void*)texture, size, { 0, 1 }, { 1, 0 });
	ImGui::End();
}


void RT_Viewport::invokeRenderProcedure() {
	int blockW = 16;
	int blockH = 16;
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

	renderImage << <blocks, threads >> > (viewCudaSurfaceObject, size.x, size.y);
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
	glDeleteTextures(1, &texture);
}
