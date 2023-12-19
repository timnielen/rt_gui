#pragma once
#include "imgui.h"
#include "Shader.h"
#include <cstdbool>
#include "Camera.h"
#include "Grid.h"
#include "Model.h"
#include "Ray.h"

struct ViewportSettings {
	bool wireframe = false;
	bool drawNormals = false;
	bool drawAABBs = false;
	float viewNormalsLength = 0.1;
};

class Viewport
{
private:
	ImVec2 size;
	Model *model;
	Grid grid;
	glm::vec4 clearColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
	unsigned int intermediateFBO = 0;
	unsigned int framebuffer = 0;
	unsigned int screenTexture = 0;
	unsigned int textureColorbuffer = 0;
	unsigned int rbo = 0;
	unsigned int axisVAO;
	float nearPlane, farPlane;
	void updateFramebuffer();

public:
	void updateCameraProjection();
	bool firstMouse = true;
	ImVec2 lastMousePos = { 0,0 };
	Camera camera;
	Shader normalsShader;
	Shader shader;
	Shader gridShader;
	Shader axisShader;
	Shader aabbShader;
	glm::vec3 dirLight = glm::vec3(1, -1, -1);
	float fov = 45.0f;
	ViewportSettings settings;
	Viewport(const Shader&);
	void setShader(const Shader&);
	void render(float deltaTime);
	~Viewport();
};

