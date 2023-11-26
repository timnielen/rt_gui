#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>

void Camera::setPosition(glm::vec3 pos) {
	position = pos;
	updateView();
}

void Camera::updateProjection(float fov, float aspect, float near, float far) {
	projection = glm::perspective(glm::radians(fov), aspect, near, far);
	//projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
}

void Camera::updateView() {
	view = glm::mat4(1);
	view = glm::translate(view, position);
}

void Camera::useInShader(Shader shader) {
	shader.setMat4("projection", projection);
	shader.setMat4("view", view);
}