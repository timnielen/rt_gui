#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>

void Camera::setPosition(glm::vec3 pos) {
	position = pos;
	updateView();
}

void Camera::setPerspective(float fov, float aspect, float near, float far) {
	projection = glm::perspective(glm::radians(fov), aspect, near, far);
}

void Camera::updateView() {
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction = glm::normalize(direction);

	glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::normalize(glm::cross(worldUp, direction));
	up = glm::cross(direction, right);

	view = glm::lookAt(position, position + direction, up);
}

void Camera::useInShader(Shader shader) {
	shader.setMat4("projection", projection);
	shader.setMat4("view", view);
}