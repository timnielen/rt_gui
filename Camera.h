#pragma once
#include <GL/gl3w.h>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Model.h"
#include "d_Model.h"
#include "GlobalTypes.h"
#include "File.h"

#include <cuda_gl_interop.h>
#include "GlobalTypes.h"
#include "cuda_helper.h"

struct cudaSurface {
	cudaGraphicsResource_t gfxRes = NULL;
	cudaSurfaceObject_t surfaceObject;
	uint glTexture;
	cudaSurface() {}
	void init(const uint& tex) {
		glTexture = tex;
		if (gfxRes != NULL)
			destroy();
		checkCudaErrors(cudaGraphicsGLRegisterImage(&gfxRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}
	void map() {
		checkCudaErrors(cudaGraphicsMapResources(1, &gfxRes));

		cudaArray_t viewCudaArray;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, gfxRes, 0, 0));
		cudaResourceDesc viewCudaArrayResourceDesc;
		{
			viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
			viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
		}
		checkCudaErrors(cudaCreateSurfaceObject(&surfaceObject, &viewCudaArrayResourceDesc));
	}
	void unmap() {
		checkCudaErrors(cudaGraphicsUnmapResources(1, &gfxRes));
	}

	void destroy() {
		std::cout << "Cuda Texture unloaded." << std::endl;
		checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
		gfxRes = NULL;
		checkCudaErrors(cudaDestroySurfaceObject(surfaceObject));
	}
};

struct cudaTexture {
	cudaGraphicsResource_t gfxRes = NULL;
	cudaTextureObject_t texObject;
	uint glTexture;
	void init(uint tex) {
		glTexture = tex;
		if (gfxRes != NULL)
			destroy();
		checkCudaErrors(cudaGraphicsGLRegisterImage(&gfxRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	}
	void map() {
		if (gfxRes == NULL) std::cout << "error!!" << std::endl;
		checkCudaErrors(cudaGraphicsMapResources(1, &gfxRes));

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaArray_t viewCudaArray;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, gfxRes, 0, 0));
		cudaResourceDesc viewCudaArrayResourceDesc;
		{
			viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
			viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
		}
		checkCudaErrors(cudaCreateTextureObject(&texObject, &viewCudaArrayResourceDesc, &texDesc, NULL));
	}
	void unmap() {
		checkCudaErrors(cudaGraphicsUnmapResources(1, &gfxRes));
	}
	void destroy() {
		std::cout << "Cuda Texture unloaded." << std::endl;
		checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
		gfxRes = NULL;
		checkCudaErrors(cudaDestroyTextureObject(texObject));
	}
};

class Camera
{
public:
	//Rasterization
	glm::vec3 position = glm::vec3(0);
	glm::vec3 right = glm::vec3(1, 0, 0);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 direction = glm::vec3(0,0,-1);
	float yaw = -90.0f;
	float pitch = 0.0f;

	float fov = 45.0f;
	float nearPlane = 0.1f;
	float farPlane = 1000.f;
	int imageWidth = 0;
	int imageHeight = 0;
	void setPosition(glm::vec3 pos);
	virtual void updateView() = 0;
	virtual void resize(int width, int height) = 0;
	virtual uint render() = 0;

};

class Rasterizer : public Camera {
public:
	Rasterizer(Model scene) : scene(scene) {

	}
	void updateView() override;
	void resize(int width, int height) override;
	uint render() override;
private:
	Model scene;
	glm::mat4 projection = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	Shader shader;
	uint intermediateFBO = 0;
	uint framebuffer = 0;
	uint screenTexture = 0;
	uint textureColorbuffer = 0;
	uint rbo = 0;
};

class RayTracer : public Camera {
public:
	RayTracer(const Model& scene) {
		this->scene = d_Model(scene).hitable;
		environment.init(load_texture("./assets/hdri/sunflowers_puresky_4k.hdr"));
	}
	__device__ Ray getRay(float u, float v) {
		auto pixel_center = pixel00_loc + (u * viewportU * right) + (v * viewportV * up);
		auto ray_direction = pixel_center - position;
		return Ray(Vec3(position), ray_direction);
	}
	void updateView() override;
	void resize(int width, int height) override;
	uint render() override;
	int samples = 1;
	int max_steps = 5;
	~RayTracer() {
		if (deviceCopy != nullptr)
			checkCudaErrors(cudaFree(deviceCopy));
		renderImage.destroy();
		environment.destroy();
	}
private:
	Hitable** scene;
	cudaTexture environment;
	cudaSurface renderImage;
	uint image;
	curandState* randomStates = nullptr;
	int blockW = 16;
	int blockH = 16;
	uint accumulation = 0;
	Vec3 pixel00_loc;
	float pixelDeltaU, pixelDeltaV, viewportU, viewportV;
	RayTracer* deviceCopy = nullptr;
};