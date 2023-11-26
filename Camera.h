#pragma once
#include <glm/glm.hpp>
#include "Shader.h"
class Camera
{
public:
	glm::mat4 projection = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::vec3 position = glm::vec3(0);
	glm::vec3 right = glm::vec3(-1, 0, 0);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 direction = glm::vec3(0);
	float yaw = -90.0f;
	float pitch = 0;
	void setPerspective(float, float, float, float);
public:
	Camera() {}
	void setPosition(glm::vec3);
	void updateView();
	void useInShader(Shader shader);
};

