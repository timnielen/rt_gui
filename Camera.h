#pragma once
#include <GL/gl3w.h>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Scene.h"
#include "GlobalTypes.h"
#include "File.h"
#include "GraphicsResource.h"
#include "cuda_helper.h"
#include <curand_kernel.h>

enum RenderType { renderTypeRasterize, renderTypeRayTrace, renderTypeCount};

class Renderer {
protected:
	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 direction;
	int imageWidth = 0;
	int imageHeight = 0;
	float fov, nearPlane, farPlane;
	virtual void updateView() = 0;
public:
	void setViewVectors(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& right, const glm::vec3& up) {
		this->position = position;
		this->direction = direction;
		this->right = right;
		this->up = up;
		updateView();
	}
	virtual uint render() = 0;
	virtual void resize(const int& width, const int& height, const float& fov, const float& nearPlane, const float& farPlane) = 0;
};

class Camera
{
private:
	Renderer* renderer[renderTypeCount];
	RenderType activeRenderer = renderTypeRasterize;
public:
	Camera(Scene& scene);
	~Camera();
	void setRenderer(RenderType renderer) {
		if(renderer >= renderTypeCount)
		{
			std::cerr << "invalid renderer type" << std::endl;
			return;
		}
		activeRenderer = renderer;
	}
	Renderer* getRenderer(RenderType type) { return renderer[type]; }
	glm::vec3 position = glm::vec3(0);
	glm::vec3 right = glm::vec3(1, 0, 0);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 direction = glm::vec3(0,0,-1);
	float distance = 5;
	glm::vec3 focalPoint = glm::vec3(0);
	float yaw = -90.0f;
	float pitch = 0.0f;

	float fov = 45.0f;
	float nearPlane = 0.1f;
	float farPlane = 1000.f;
	void setPosition(glm::vec3 pos);
	void update();
	void resize(const int& width, const int& height);
	uint render();
};


class Rasterizer : public Renderer {
public:
	Rasterizer(Scene& scene) : scene(scene), shader("./shader/vertex.glsl", "./shader/fragment.glsl") {
		normalsShader.setVertex("./shader/vertex.glsl");
		normalsShader.setGeometry("./shader/Normals/normals_geometry.glsl");
		normalsShader.setFragment("./shader/Normals/in_color_fragment.glsl");
		normalsShader.link();

		aabbShader.setVertex("./shader/AABB/vertex.glsl");
		aabbShader.setGeometry("./shader/AABB/geometry.glsl");
		aabbShader.setFragment("./shader/AABB/fragment.glsl");
		aabbShader.link();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_MULTISAMPLE);
	}
	void updateView() override;
	void resize(const int& width, const int& height, const float& fov, const float& nearPlane, const float& farPlane) override;
	uint render() override;
	bool wireframe = false;
	bool showNormals = false;
	bool showAABBs = false;
	int aabbDepth = 0;
	float normalsLength = 0.1f;
private:
	Scene& scene;
	glm::mat4 projection = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::vec4 clearColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
	Shader shader, normalsShader, aabbShader;
	uint intermediateFBO = 0;
	uint framebuffer = 0;
	uint screenTexture = 0;
	uint textureColorbuffer = 0;
	uint rbo = 0;
};

class RayTracer : public Renderer {
public:
	RayTracer(const Scene& scene) {
		this->scene = scene.hitable;
		this->lights = scene.lights;
		this->lightCount = scene.lightCount;
		environment.init(load_texture("./assets/hdri/sunflowers_puresky_4k.hdr")); 
	}
	__device__ Ray getRay(float u, float v) {
		auto pixel_center = pixel00_loc + (u * viewportU * right) + (v * viewportV * up);
		auto ray_direction = pixel_center - position;
		return Ray(Vec3(position), ray_direction);
	}
	void updateView() override;
	void resize(const int& width, const int& height, const float& fov, const float& nearPlane, const float& farPlane) override;
	uint render() override;
	int samples = 1;
	int max_steps = 5;
	~RayTracer() {
		if (deviceCopy != nullptr)
			checkCudaErrors(cudaFree(deviceCopy));
		renderImage.destroy();
		environment.destroy();
	}
private:
	Hitable** scene;
	Hitable** lights;
	int lightCount;
	cudaTexture environment;
	cudaSurface renderImage;
	uint image;
	curandState* randomStates = nullptr;
	int blockW = 16;
	int blockH = 16;
	uint accumulation = 0;
	Vec3 pixel00_loc;
	float pixelDeltaU, pixelDeltaV, viewportU, viewportV;
	RayTracer* deviceCopy = nullptr;
};
