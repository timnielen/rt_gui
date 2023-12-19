#pragma once
#include "Vec3.h"
#include "Ray.h"

class Material;

class HitRecord {
public:
	Point3 p;
	Vec3 normal;
	float t;
	bool front_face;
	Material* mat;

	__device__ __host__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
		// Sets the hit record normal vector.
		// NOTE: the parameter `outward_normal` is assumed to have unit length.

		front_face = dot(r.direction, outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hitable {
public:
	__device__ __host__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const = 0;
};

class HitableList : public Hitable {
public:
	__device__ __host__ HitableList(Hitable** list, unsigned int length) : list(list), list_length(length) {}
	__device__ __host__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
		HitRecord temp_rec;
		bool hit_anything = false;
		float closest_so_far = tmax;
		for (int i = 0; i < list_length; i++) {
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
};