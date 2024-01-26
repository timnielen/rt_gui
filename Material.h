#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <assimp/material.h>

struct HitRecord;

#include "Hit.h"
#include "Ray.h"
#include "Vec3.h"
#include "GlobalTypes.h"
#include <vector>


__device__ inline Vec3 random_in_unit_sphere(curandState* local_rand_state) {
	float2 rand = curand_normal2(local_rand_state);
	return Vec3(rand.x, rand.y, curand_normal(local_rand_state));
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

__device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - 2 * dot(v, n) * n;
}

__device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
	float cos_theta = fminf(dot(-uv, n), 1.0f);
	Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

class Material {
public:
    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material {
    Vec3 albedo;
public:
    __device__ __host__ Lambertian(const Vec3& albedo) : albedo(albedo) {}

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

class Dielectric : public Material {
public:
	__device__ Dielectric(float index_of_refraction) : ir(index_of_refraction) {}

	__device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state)
		const override {
		attenuation = Vec3(1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

		Vec3 unit_direction = d_normalize(r_in.direction);

		float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

		Vec3 direction;
		if (refraction_ratio * sin_theta > 1.0f || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
			// Must Reflect
			direction = reflect(d_normalize(r_in.direction), rec.normal);
		}
		else {
			// Can Refract
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = Ray(rec.p, direction);
		return true;
	}

private:
	float ir; // Index of Refraction

	__device__ static float reflectance(float cosine, float ref_idx) {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * powf((1 - cosine), 5);
	}
};

enum TextureType { textureTypeDiffuse = 0, textureTypeSpecular, textureTypeNormal, textureTypeCount};

struct TextureStack {
	Vec3 baseColor;
	std::vector<uint> texIndices;
	std::vector<float> texBlend;
};

class MultiMaterial {
public:
	MultiMaterial() : index(-1) {
		textures[textureTypeDiffuse].baseColor = Vec3(1.0f, 0.1f, 0.5f);
		textures[textureTypeSpecular].baseColor = Vec3(0.1f);
	}
	MultiMaterial(int index) : index(index) {}
	std::string name = "Default Material";
	int index;
	aiBlendMode blendMode = aiBlendMode_Default;
	float opacity = 1.0f;
	float shininess = 32.0f;
	float shininessStrength = 1.0f;
	float refractionIndex = 0.0f;
	TextureStack textures[textureTypeCount];
};

const MultiMaterial DEFAULT_MATERIAL;