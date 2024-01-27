#pragma once
#include "imgui.h"
#include "Shader.h"
#include "DynamicViewport.h"

class App {
private:
	DynamicViewport* viewport;
public:
	App();
	void showTextureStack(MultiMaterial& mat, const char* name, TextureType type);
	void renderUI(ImGuiIO& io);
	~App();
};
