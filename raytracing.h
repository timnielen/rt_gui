#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RT_Camera.h"
#include "Material.h"
#include <curand_kernel.h>


__global__ void init_scene(Hitable** scene, Hitable** objects) {
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;
	objects[0] = new Sphere(Vec3(0, 0, -1), 0.5f, new Lambertian(Vec3(0.5f, 0.2f, 0.1f)));
	objects[1] = new Sphere(Vec3(0, -100.5f, -1), 100, new Lambertian(Vec3(0.3f)));

	*scene = new HitableList(objects, 2);
}

__global__ void free_scene(Hitable** scene, Hitable** objects, unsigned int sizeObjects) {
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;
	for (unsigned int i = 0; i < sizeObjects; i++) {
		delete ((Sphere*)objects[i])->mat;
		delete objects[i];
	}
	delete* scene;
}

__device__ Vec3 ray_color(Ray& r, Hitable* obj, curandState* local_rand_state, int max_steps) {
	Ray cur_ray = r;
	Vec3 cur_attenuation = 1.0f;
	for (int i = 0; i < max_steps; i++) {
		HitRecord rec;
		if (obj->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			Ray scatter;
			Vec3 attenuation;
			if (!rec.mat->scatter(cur_ray, rec, attenuation, scatter, local_rand_state))
				return Vec3(0.0f);
			cur_attenuation = cur_attenuation * attenuation;
			cur_ray = scatter;
		}
		else {
			Vec3 unit_direction = d_normalize(cur_ray.direction);
			float t = 0.5f * (unit_direction.y + 1.0f);
			Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return Vec3(0.0f); // exceeded recursion
}

__global__ void renderInit(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(6969, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void renderImage(cudaSurfaceObject_t surface, int max_x, int max_y, RT_Camera *cam, Hitable** scene, curandState* rand_state, int samples, int max_steps, int accumulation) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;
	curandState* local_rand_state = &rand_state[pixel_index];

	Vec3 col(0, 0, 0);
	for (int s = 0; s < samples; s++) {
		float u = float(i + curand_uniform(local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(local_rand_state)) / float(max_y);
		Ray r = cam->getRay(u,v);
		col += ray_color(r, *scene, local_rand_state, max_steps);
	}
	col /= (float)samples;
	//Vec3 col = Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
	//auto col = Vec3(0.4f, 0.3f, 1.0f);
	float4 prev_col;
	surf2Dread(&prev_col, surface, i * sizeof(float4), j);
	Vec3 prev = Vec3(prev_col.x, prev_col.y, prev_col.z);
	
	//reverse Gamma correction
	prev = prev * prev;
	prev *= (accumulation - 1);
	col += prev;
	col /= accumulation;
	surf2Dwrite(col.toGamma().toColor(), surface, i * sizeof(float4), j);
}