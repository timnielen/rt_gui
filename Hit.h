#pragma once
#include "Vec3.h"
#include "Ray.h"
#include "AABB.h"

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
	AABB aabb;
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const = 0;
	__host__ virtual Hitable* toGPU() = 0;
};

__device__ unsigned int expandBits(unsigned int v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(Vec3 v);

class HitableList : public Hitable {
public:
	__device__ __host__ HitableList(Hitable** list, unsigned int length) : list(list), list_length(length) {
		if (list_length == 0) return;
		aabb = list[0]->aabb;
		for (unsigned int i = 1; i < list_length; i++) {
			aabb = AABB(aabb, list[i]->aabb);
		}
	}

	__host__ Hitable* toGPU() override {
		Hitable** temp;
		cudaMallocManaged((void**)&temp, list_length * sizeof(Hitable*));
		for (int i = 0; i < list_length; i++) {
			temp[i] = list[i]->toGPU();
			delete list[i];
			list[i] = nullptr;
		}
		//delete[] list;
		list = temp;

		HitableList* gpu;
		cudaMalloc((void**)&gpu, sizeof(HitableList));
		cudaMemcpy(gpu, this, sizeof(HitableList), cudaMemcpyHostToDevice);
		return gpu;
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


	__device__ void genMortonCodes() {
		mortonCodes = new unsigned int[list_length];
		sortedIndices = new unsigned int[list_length];
		Vec3 dimensions = aabb.max - aabb.min;
		for (unsigned int i = 0; i < list_length; i++) {
			sortedIndices[i] = i;
			AABB child = list[i]->aabb;
			Vec3 centroid = (child.min + child.max) / 2;
			centroid = centroid - aabb.min;
			centroid /= dimensions;
			mortonCodes[i] = morton3D(centroid);

		}
	}

	Hitable** list;
	unsigned int* mortonCodes;
	unsigned int* sortedIndices;
	unsigned int list_length;
private:

};