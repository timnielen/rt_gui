#pragma once
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
	__host__ void init(uint tex) {
		glTexture = tex;
		if (gfxRes != NULL)
			destroy();
		checkCudaErrors(cudaGraphicsGLRegisterImage(&gfxRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	}
	__host__ void map() {
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
	__host__ void unmap() {
		checkCudaErrors(cudaGraphicsUnmapResources(1, &gfxRes));
	}
	__host__ void destroy() {
		std::cout << "Cuda Texture unloaded." << std::endl;
		checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
		gfxRes = NULL;
		checkCudaErrors(cudaDestroyTextureObject(texObject));
	}
};
