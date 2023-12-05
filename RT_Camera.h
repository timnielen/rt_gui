#pragma once
#include "Ray.h"
class RT_Camera {
public:
	__device__ __host__ RT_Camera() {
	};
	Vec3 pixel00_loc, camera_center, pixel_delta_u, pixel_delta_v, viewport_u, viewport_v;
	__device__ __host__ void update(const int& x, const int& y);
	__device__ __host__ Ray getRay(float u, float v) {
		auto pixel_center = pixel00_loc + (u * viewport_u) + (v * viewport_v);
		auto ray_direction = pixel_center - camera_center;
		return Ray(camera_center, ray_direction);
	}
};
