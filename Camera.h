#pragma once
#include <glm/glm.hpp>
#include "Shader.h"
#include "Ray.h"
class Camera
{
public:
	//Rasterization
	glm::mat4 projection = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::vec3 position = glm::vec3(0, 0, 3);
	glm::vec3 right = glm::vec3(1, 0, 0);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 direction = glm::vec3(0,0,-1);
	float yaw = -90.0f;
	float pitch = 0.0f;
	void setProjection(float fov, float width, float height, float near, float far = 1000.0f);

	//Raytracing
	Vec3 pixel00_loc;
	float pixelDeltaU, pixelDeltaV, focalLength, viewportU, viewportV;

	Camera() {}
	void setPosition(glm::vec3);
	void updateView();
	void useInShader(Shader shader);

	//Raytracing
	__device__ Ray getRay(float u, float v) {
		auto pixel_center = pixel00_loc + (u * viewportU * right) + (v * viewportV * up);
		auto ray_direction = pixel_center - position;
		return Ray(Vec3(position), ray_direction);
	}
};

