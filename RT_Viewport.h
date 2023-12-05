#pragma once
#include <imgui.h>
#include <GL/gl3w.h>
#include <cuda_gl_interop.h>
#include "Vec3.h"
#include "Ray.h"
#include "Hit.h"
#include "RT_Camera.h"

#include <curand_kernel.h>

class RT_Viewport {
public:
	RT_Viewport();
	~RT_Viewport();
	void render(float deltaTime);
	void invokeRenderProcedure();
	RT_Camera *cam;
	Hitable** objects;
	Hitable** scene;
	int samples = 1;
	int max_steps = 5;
	
private:
	int blockW = 8;
	int blockH = 8;
	curandState* d_rand_state = nullptr;
	unsigned int texture;
	ImVec2 size;
	cudaGraphicsResource_t gfxRes;
	void updateFramebuffer();
};