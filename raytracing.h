#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RT_Camera.h"
#include "Material.h"
#include <curand_kernel.h>

__device__
float2 sampleSphericalMap(Vec3 direction)
{
	float u = atan2f(direction.z, direction.x) * 0.1591f + 0.5f;
	float v = 0.5f - asinf(direction.y) * 0.3183f;
	return make_float2(u, v);
}

__global__ void init_scene(Hitable** scene, Hitable** objects) {
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;
	objects[0] = new Sphere(Vec3(0, -100.5f, -1), 100, new Lambertian(Vec3(0.3f)));
	objects[1] = new Sphere(Vec3(-1, 0, -1), 0.5f, new Metal(Vec3(1), 0.0f));
	objects[2] = new Sphere(Vec3(1, 0, -1), 0.5f, new Metal(Vec3(1.0f, 0.2f, 0.1f), 0.5f));
	objects[3] = new Sphere(Vec3(0, 0, -1), 0.5f, new Lambertian(Vec3(0.2f, 1.0f, 0.5f)));


	*scene = new HitableList(objects, 4);
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

__device__ Vec3 ray_color(Ray& r, Hitable* obj, curandState* local_rand_state, int max_steps, cudaTextureObject_t hdri) {
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
			/*Vec3 unit_direction = d_normalize(cur_ray.direction);
			float t = 0.5f * (unit_direction.y + 1.0f);
			Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);*/

			float2 hdriUV = sampleSphericalMap(d_normalize(cur_ray.direction));
			float4 col_hdri = tex2D<float4>(hdri, hdriUV.x, hdriUV.y);
			Vec3 c = Vec3(col_hdri.x, col_hdri.y, col_hdri.z);
			//reverse Gamma correction
			c = c * c;
			return cur_attenuation * c;
		}
	}
	return Vec3(0.0f); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(6969, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_image(cudaSurfaceObject_t surface, int max_x, int max_y, RT_Camera *cam, Hitable** scene, cudaTextureObject_t hdri, curandState* rand_state, int samples, int max_steps, int accumulation) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	float u = float(i / float(max_x));
	float v = float(j / float(max_y));
	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;
	curandState* local_rand_state = &rand_state[pixel_index];

	Vec3 col(0, 0, 0);
	for (int s = 0; s < samples; s++) {
		float u = float(i + curand_uniform(local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(local_rand_state)) / float(max_y);
		Ray r = cam->getRay(u,v);
		col += ray_color(r, *scene, local_rand_state, max_steps, hdri);
		
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
	
	//surf2Dwrite(color, surface, i * sizeof(float4), j);

}