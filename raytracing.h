#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Camera.h"
#include "Material.h"
#include <curand_kernel.h>
#include "BVH.h"
#include "d_Model.h"

__device__
float2 sampleSphericalMap(Vec3 direction)
{
	float u = atan2f(direction.z, direction.x) * 0.1591f + 0.5f;
	float v = 0.5f - asinf(direction.y) * 0.3183f;
	return make_float2(u, v);
}

__device__ bool compare(int a, int b) {
	return a <= b;
}

__global__ void init_scene(Hitable** scene, Hitable** hlist) {
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	/*d_Vertex *verts = new d_Vertex[3];
	verts[0].Position = Vec3(-0.5f, 0.0f, 0.0f);
	verts[1].Position = Vec3(0.5f, 0.0f, 0.1f);
	verts[2].Position = Vec3(0.0f, 1.0f, -0.5f);
	verts[0].Normal = Vec3(0.0f, 0.0f, 1.0f);
	verts[1].Normal = Vec3(0.0f, 0.0f, 1.0f);
	verts[2].Normal = Vec3(0.0f, 0.0f, 1.0f);
	objects[3] = new Triangle(0, 1, 2, verts, new Metal(Vec3(1.0f), 0.0f));*/

	curandState local_rand_state;
	curand_init(6969, 0, 0, &local_rand_state);
	//HitableList* hlist = new HitableList(objects, objectCount);
	*scene = *hlist; // new BVH_Node(objects, 0, 10, &local_rand_state);
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
			//return Vec3(1.0f);
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
			//Vec3 c = Vec3(0.2f);
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
	curand_init(6969 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render_image(cudaSurfaceObject_t surface, int max_x, int max_y, RayTracer* cam, Hitable** scene, cudaTextureObject_t hdri, curandState* rand_state, int samples, int max_steps, int accumulation) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;
	curandState* local_rand_state = &rand_state[pixel_index];
	//printf("%d %d: %p\n", i, j, *scene);
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
}