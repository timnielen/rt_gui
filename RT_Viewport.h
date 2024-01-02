#pragma once
#include <imgui.h>
#include <GL/gl3w.h>
#include <cuda_gl_interop.h>
#include "Vec3.h"
#include "Ray.h"
#include "Hit.h"
#include "Camera.h"

#include <curand_kernel.h>

struct ImageResource {
	cudaGraphicsResource_t gfxRes = NULL;
	cudaSurfaceObject_t viewCudaSurfaceObject;
	cudaTextureObject_t viewCudaTextureObject;
	unsigned int texture;
	void init(unsigned int tex, int flags = cudaGraphicsRegisterFlagsSurfaceLoadStore);
	void mapTexture();
	void mapSurface();
	void unmap();
	void destroy();
	cudaSurfaceObject_t getSurface() const {
		return viewCudaSurfaceObject;
	}
	cudaTextureObject_t getTexture() const {
		return viewCudaTextureObject;
	}
};

class RT_Viewport {
public:
	RT_Viewport();
	~RT_Viewport();
	void render(float deltaTime);
	void invokeRenderProcedure();
	Camera *camera;
	Hitable** objects;
	Hitable** scene;
	int samples = 1;
	int max_steps = 5;
	
private:
	bool firstMouse = true;
	ImVec2 lastMousePos = { 0,0 };
	bool resizeFinished = false;
	int blockW = 16;
	int blockH = 16;
	unsigned int accumulation = 0;
	curandState* d_rand_state = nullptr;
	unsigned int texture;
	ImVec2 size;
	ImageResource renderedImage;
	ImageResource hdri;
	bool updateFramebuffer();
};