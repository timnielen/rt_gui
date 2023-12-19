#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

struct HitRecord;

#include "Hit.h"
#include "Ray.h"
#include "Vec3.h"

#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ inline Vec3 random_in_unit_sphere(curandState* local_rand_state) {
	Vec3 p;
	do {
		p = 2.0f * RANDVEC3 - 1.0f;
	} while (p.length_squared() >= 1.0f);
	return p;
}

__device__ inline Vec3 random_unit_vector(curandState* local_rand_state) {
	return d_normalize(random_in_unit_sphere(local_rand_state));
}

__device__ inline Vec3 random_on_hemisphere(const Vec3& normal, curandState* local_rand_state) {
	Vec3 on_unit_sphere = random_unit_vector(local_rand_state);
	if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return on_unit_sphere;
	else
		return -on_unit_sphere;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - 2 * dot(v, n) * n;
}

class Material {
public:
    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material {
    Vec3 albedo;
public:
    __device__ Lambertian(const Vec3& albedo) : albedo(albedo) {}

	__device__ bool scatter(
		const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override {
		Vec3 dir = rec.normal + random_on_hemisphere(rec.normal, local_rand_state);
		
		// Catch degenerate scatter direction
		if (dir.near_zero())
			dir = rec.normal;

		attenuation = albedo;
		scattered = Ray(rec.p, dir);
		return true;
	}
};

class Metal : public Material {
public:
	__device__ Metal(const Vec3& a, float roughness) : albedo(a), roughness(roughness) {}

	__device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state)
		const override {
		Vec3 reflected = reflect(d_normalize(r_in.direction), rec.normal);
		scattered = Ray(rec.p, reflected + roughness * random_unit_vector(local_rand_state));
		attenuation = albedo;
		return dot(scattered.direction, rec.normal) > 0;
	}

private:
	Vec3 albedo;
	float roughness;
};