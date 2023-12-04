#pragma once
#include <imgui.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL/gl3w.h>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>

// Includes CUDA
#include <cuda_runtime_api.h>

class RT_Viewport {
public:
	RT_Viewport();
	~RT_Viewport();
	void render(float deltaTime);
	void invokeRenderProcedure();
	
private:
	unsigned int texture;
	ImVec2 size;
	cudaGraphicsResource_t gfxRes;
	void updateFramebuffer();
};