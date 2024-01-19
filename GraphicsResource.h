#pragma once
#include <cuda_gl_interop.h>
#include "GlobalTypes.h"
#include "cuda_helper.h"

struct cudaSurface {
	cudaGraphicsResource_t gfxRes = NULL;
	cudaSurfaceObject_t surfaceObject;
	uint glTexture;
	cudaSurface() {}
	cudaSurface(const uint& tex) : glTexture(tex) {
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

	~cudaSurface() {
		checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
		gfxRes = NULL;
		checkCudaErrors(cudaDestroySurfaceObject(surfaceObject));
	}
};

struct cudaTexture {
	cudaGraphicsResource_t gfxRes = NULL;
	cudaTextureObject_t texObject;
	uint glTexture;
	cudaTexture() {}
	cudaTexture(uint tex) : glTexture(tex) {
		checkCudaErrors(cudaGraphicsGLRegisterImage(&gfxRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	}
	void map() {
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
	~cudaTexture() {
		checkCudaErrors(cudaGraphicsUnregisterResource(gfxRes));
		gfxRes = NULL;
		checkCudaErrors(cudaDestroyTextureObject(texObject));
	}
};