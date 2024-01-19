#pragma once
#include "imgui.h"
#include "Camera.h"
#include "Shader.h"
#include <cstdbool>
#include "Model.h"
#include "d_Model.h"
#include "Ray.h"


class DynamicViewport
{
private:
	Camera* camera;
	ImVec2 size;
	Model scene;
	void updateFramebuffer();
	void handleUserInput(float deltaTime);
	float mouseSensitivity = 0.1f;
	bool firstMouse = true;
	ImVec2 lastMousePos = { 0,0 };

public:	
	DynamicViewport();
	~DynamicViewport();
	void draw(float deltaTime);
};

