#pragma once
#include "imgui.h"
#include "Camera.h"
#include "Shader.h"
#include <cstdbool>
#include "Scene.h"
#include "Ray.h"

struct ViewportSettings {
	uint renderer = renderTypeRasterize;
};


class DynamicViewport
{
private:
	ImVec2 size;
	void updateFramebuffer();
	void handleUserInput(float deltaTime);
	float mouseSensitivity = 0.1f;
	bool firstMouse = true;
	ImVec2 lastMousePos = { 0,0 };

public:
	Camera* camera;
	Scene scene;
	ViewportSettings settings;
	void updateSettings() {
		camera->setRenderer((RenderType)settings.renderer);
	}
	DynamicViewport();
	~DynamicViewport();
	void draw(float deltaTime);
};

