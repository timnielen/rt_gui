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

__global__
void add(int n, float* x, float* y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;
	y[i] = x[i] + y[i];
}


RT_Viewport::RT_Viewport() : size({ -1,-1 }) {
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
	//checkCudaErrors(cudaSetDevice(devices[0]));


	glGenTextures(1, &texture);
	checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(RT_Camera)));

	checkCudaErrors(cudaMalloc((void**)&objects, 4*sizeof(Hitable*)));
	checkCudaErrors(cudaMalloc((void**)&scene, sizeof(Hitable*)));
	init_scene<<<1, 1 >>>(scene, objects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	
}


float4* data = nullptr;
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

	if (data != nullptr) {
		cudaFree(data); 
	}
	cudaMallocManaged(&data, sizeof(float4) * size.x * size.y);
	for (int i = 0; i < size.x * size.y; i++)
		data[i] = make_float4(0.1f, 0.5f, 1.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, texture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, data);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	renderedImage.init(texture);

	//init randoms
	if(d_rand_state != nullptr)
		checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, size.x * size.y * sizeof(curandState)));

	dim3 blocks(size.x / blockW + 1, size.y / blockH + 1);
	dim3 threads(blockW, blockH);
	render_init << <blocks, threads >> > (size.x, size.y, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	accumulation = 0;

	resizeFinished = true;
	return true;
}

bool first = true;
void RT_Viewport::render(float deltaTime) {
	ImGui::Begin("RT Viewport");
	bool fbDrawable = updateFramebuffer();
	if (size.x <= 0 || size.y <= 0) {
		ImGui::End();
		return;
	}
	if (fbDrawable && !first) {
		invokeRenderProcedure();
	}
	first = false;

	ImGui::Image((void*)texture, size); // , { 0, 1 }, { 1, 0 });
	ImGui::End();
}


void RT_Viewport::invokeRenderProcedure() {
	// Render our buffer
	dim3 blocks(size.x / blockW + 1, size.y / blockH + 1);
	dim3 threads(blockW, blockH);

	accumulation++;
	renderedImage.map();
	render_image<<<blocks, threads>>> (renderedImage.getSurface(), size.x, size.y, cam, scene, d_rand_state, samples, max_steps, accumulation);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	renderedImage.unmap();

	checkCudaErrors(cudaStreamSynchronize(0));
}

RT_Viewport::~RT_Viewport()
{
	cudaFree(data);
	renderedImage.destroy();
	checkCudaErrors(cudaDeviceSynchronize());
	free_scene<<<1, 1 >>>(scene, objects, 4);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(objects));
	checkCudaErrors(cudaFree(scene));
	checkCudaErrors(cudaFree(cam));
	glDeleteTextures(1, &texture);
}

void ImageResource::init(unsigned int texture) {
	if (gfxRes != NULL)
		destroy();
	auto e = cudaGraphicsGLRegisterImage(&gfxRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	checkCudaErrors(e);
}

void ImageResource::map() {

	checkCudaErrors(cudaGraphicsMapResources(1, &gfxRes));

	cudaArray_t viewCudaArray;
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, gfxRes, 0, 0));
	cudaResourceDesc viewCudaArrayResourceDesc;
	{
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
	}
	checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
}

void ImageResource::unmap() {
	checkCudaErrors(cudaGraphicsUnmapResources(1, &gfxRes));
}

void ImageResource::destroy() {
	checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
	gfxRes = NULL;
	checkCudaErrors(cudaDestroySurfaceObject(viewCudaSurfaceObject));
}