#pragma once
#include "imgui.h"
#include "Viewport.h"
#include "Shader.h"
class App {
private:
	Viewport* viewport;
	Shader shader;
public:
	App();
	void renderUI(ImGuiIO& io);
	~App();
};
