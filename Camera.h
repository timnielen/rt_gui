#pragma once
#include <glm/glm.hpp>
#include "Shader.h"
class Camera
{
public:
	glm::mat4 projection = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::vec3 position = glm::vec3(0);
	void updateProjection(float, float, float, float);
public:
	Camera() {}
	void setPosition(glm::vec3);
	void updateView();
	void useInShader(Shader shader);
};
