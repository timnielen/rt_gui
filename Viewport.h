#pragma once
#include "imgui.h"
#include "Shader.h"
#include <cstdbool>
#include "Object3D.h"
#include "Camera.h"
struct ViewportSettings {
	bool wireframe = false;
};

class Viewport
{
private:
	ImVec2 size;
	Object3D object;
	Camera camera;
	unsigned int framebuffer = 0;
	unsigned int textureColorbuffer = 0;
	unsigned int rbo = 0;
	void updateFramebuffer();
public:
	Shader shader;
	ViewportSettings settings;
	Viewport(const Shader&);
	void setShader(const Shader&);
	void render();
	~Viewport();
};

