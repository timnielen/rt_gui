#pragma once
#include "imgui.h"
#include "Viewport.h"
#include "Shader.h"
#include "RT_Viewport.h"

class App {
private:
	RT_Viewport* rt_viewport;
	Viewport* viewport;
	Shader shader;
public:
	App();
	void renderUI(ImGuiIO& io);
	~App();
};
