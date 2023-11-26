#pragma once
#include "Shader.h"
#include <glm/glm.hpp>
class Object3D
{
private:
	unsigned int VAO = 0;
	unsigned int texture = 0;
	glm::vec3 position = glm::vec3(0);
	glm::vec3 axis = glm::vec3(0);
	float angle = 0;
	glm::vec3 scale = glm::vec3(1.0f);
	glm::mat4 transform = glm::mat4(1);
	void updateTranformationMatrix();
public:
	Object3D();
	void setPosition(glm::vec3);
	void setRotationAxis(glm::vec3);
	void setRotationAngle(float);
	void setScale(float);
	void setScale(glm::vec3);
	void render(const Shader&);
};

