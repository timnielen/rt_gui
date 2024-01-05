#pragma once
#include <curand_kernel.h>

#include "AABB.h"
#include "Hit.h"
#include "algorithm"
#include "sort.h"


class BVH_Node : public Hitable {
	Hitable* left;
	Hitable* right;
public:
    __device__ BVH_Node(HitableList* list, curandState* local_rand_state) : BVH_Node(list->list, 0, list->size(), local_rand_state) {}
    __device__ BVH_Node(Hitable** objects, int start, int end, curandState* local_rand_state) {
        int axis = ceilf(3*curand_uniform(local_rand_state));
        auto comparator = (axis == 1) ? box_x_compare
            : (axis == 2) ? box_y_compare
            : box_z_compare;
        //auto comparator = box_x_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        }
        else if (object_span == 2) {
            if (comparator(objects[start], objects[start + 1])) {
                left = objects[start];
                right = objects[start + 1];
            }
            else {
                left = objects[start + 1];
                right = objects[start];
            }
        }
        else {
            sort::quickSort(objects, start, end-1, comparator);

            auto mid = start + object_span / 2;
            left = new BVH_Node(objects, start, mid, local_rand_state);
            right = new BVH_Node(objects, mid, end, local_rand_state);
        }
        aabb = AABB(left->aabb, right->aabb);
    }
	__device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
		if (!aabb.hit(r, tmin, tmax))
			return false;

		bool hit_left = left->hit(r, tmin, tmax, rec);
		bool hit_right = right->hit(r, tmin, hit_left ? rec.t : tmax, rec);

		return hit_left || hit_right;

	}

private:
    __device__ static bool box_compare(const Hitable* a, const Hitable* b, int axis_index) {
        return a->aabb.min[axis_index] < b->aabb.min[axis_index];
    }

    __device__ static bool box_x_compare(const Hitable* a, const Hitable* b) {
        return box_compare(a, b, 0);
    }

    __device__ static bool box_y_compare(const Hitable* a, const Hitable* b) {
        return box_compare(a, b, 1);
    }

    __device__ static bool box_z_compare(const Hitable* a, const Hitable* b) {
        return box_compare(a, b, 2);
    }
};