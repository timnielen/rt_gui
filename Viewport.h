#pragma once
#include "imgui.h"
#include "Shader.h"
#include <cstdbool>
struct ViewportSettings {
	bool wireframe = false;
};

class Viewport
{
private:
	ImVec2 size;
	unsigned int VAO = 0;
	unsigned int texture = 0;
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

