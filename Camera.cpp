#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>

void Camera::setPosition(glm::vec3 pos) {
	position = pos;
	updateView();
}

void Camera::setProjection(float fov, float width, float height, float near, float far) {
	float aspect = width / height;
	projection = glm::perspective(glm::radians(fov), aspect, near, far);

	viewportU = -2 * near * glm::tan(glm::radians(fov) / 2.0f);
	viewportV = viewportU / aspect;
	pixelDeltaU = viewportU / width;
	pixelDeltaV = viewportV / height;
	focalLength = near;
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

	// Calculate the location of the upper left pixel.
	Vec3 upperLeft = position + (direction * focalLength) - (right * viewportU / 2) - (up * viewportV / 2);
	pixel00_loc = upperLeft + 0.5 * (pixelDeltaU * right + pixelDeltaV * up);
}

void Camera::useInShader(Shader shader) {
	shader.setMat4("projection", projection);
	shader.setMat4("view", view);
}