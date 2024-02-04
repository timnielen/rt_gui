#pragma once
#include "Vec3.h"
#include "Ray.h"
#include "AABB.h"
#include <curand_kernel.h>

class Material;

class HitRecord {
public:
	Point3 p;
	Vec3 normal;
	Vec3 tangent;
	Vec3 bitangent;
	float t;
	bool front_face;
	Material* mat;
	glm::vec2 uvCoords = { 0.0f, 0.0f };

	__device__ __host__ void set_face_normal(const Ray& r, const Vec3& outward_normal, const Vec3& faceNormal) {
		// Sets the hit record normal vector.
		float d = dot(r.direction, faceNormal);
		front_face = d < 0;
		normal = front_face ? outward_normal : -outward_normal;
		tangent = front_face ? tangent : -tangent;
		bitangent = front_face ? bitangent : -bitangent;
	}

	__device__ __host__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
		set_face_normal(r, outward_normal, outward_normal);
	}
};

class Hitable {
public:
	AABB aabb;
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const = 0;
	__device__ virtual Vec3 random(curandState* local_rand_state) {
		return Vec3(0);
	}
	__device__ virtual float pdf(const Ray& r) {
		return 0;
	}
};



class HitableList : public Hitable {
public:
	__device__ HitableList(Hitable** list, const unsigned int& length) : list(list), list_length(length) {
		if (list_length == 0) return;
		aabb = list[0]->aabb;
		for (unsigned int i = 1; i < list_length; i++) {
			aabb = AABB(aabb, list[i]->aabb);
		}
	}

	__device__ HitableList(Hitable** list, const unsigned int& length, const AABB& aabb) : list(list), list_length(length) {
		if (list_length == 0) return;
		this->aabb = aabb;
	}

	__device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
		if (!aabb.hit(r, tmin, tmax))
			return false;
		HitRecord temp_rec;
		bool hit_anything = false;
		float closest_so_far = tmax;
		for (unsigned int i = 0; i < size(); i++) {
			if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}
	__device__ __host__ unsigned int size() const {
		return list_length;
	}

	Hitable** list;
	unsigned int list_length;
private:

};