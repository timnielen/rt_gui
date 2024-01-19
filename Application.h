#pragma once
#include "imgui.h"
#include "Shader.h"
#include "DynamicViewport.h"

class App {
private:
	DynamicViewport* viewport;
public:
	App();
	void renderUI(ImGuiIO& io);
	~App();
};
