#pragma once
#include <curand_kernel.h>
#include "AABB.h"
#include "Hit.h"
#include "algorithm"
#include "sort.h"

class BVH_Node : public Hitable {
public:
    Hitable *left = nullptr;
    Hitable *right = nullptr;
    bool leftIsLeaf = true;
    bool rightIsLeaf = true;
    __device__ BVH_Node() {}
    __device__ BVH_Node(Hitable* left, Hitable* right, bool leftIsLeaf, bool rightIsLeaf) : left(left), right(right), leftIsLeaf(leftIsLeaf), rightIsLeaf(rightIsLeaf) {}
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
        return false;
    }
};

class BVH : public Hitable {
public:
    BVH_Node* nodes;
    Hitable** leaves;
    unsigned int countLeaves;

    unsigned int* mortonCodes;
    unsigned int* sortedIndices;

    __device__ BVH(const HitableList& list) {
        countLeaves = list.size();
        leaves = list.list;
        nodes = new BVH_Node[countLeaves - 1];
        aabb = list.aabb;
        genMortonCodes();
    }
    
    __device__ void genMortonCodes();
	__device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override;

    // Get common prefix Length of the morton codes of the sorted leaves at indexA and indexB
    __device__ int prefixLength(unsigned int indexA, unsigned int indexB);

    __device__ void print() {
		// Allocate traversal stack from thread-local memory,
		// and push NULL to indicate that there are no postponed nodes.
		BVH_Node* stack[32];
		BVH_Node** stackPtr = stack;
		*stackPtr++ = NULL; // push

        for (int i = 0; i < countLeaves - 1; i++)
            printf("%d: %p, %p %d %p %d\n", i, nodes + i, nodes[i].left, nodes[i].leftIsLeaf, nodes[i].right, nodes[i].rightIsLeaf);
		//// Traverse nodes starting from the root.
		BVH_Node* node = nodes;
		do
		{
			Hitable* childL = node->left;
			Hitable* childR = node->right;
			printf("%p: %p %p\n", node, childL, childR);


			if (node->leftIsLeaf)
			{
				printf("%p: leaf\n", childL);
			}

			if (node->rightIsLeaf)
			{
				printf("%p: leaf\n", childR);
			}
			bool traverseL = (!node->leftIsLeaf);
			bool traverseR = (!node->rightIsLeaf);

			if (!traverseL && !traverseR)
				node = *--stackPtr; // pop
			else
			{
				node = (traverseL) ? (BVH_Node*)childL : (BVH_Node*)childR;
				if (traverseL && traverseR)
					*stackPtr++ = (BVH_Node*)childR; // push
			}
		} while (node != NULL);
    }
};

__global__ void constructBVH(BVH* bvh);
//class BVH_Node : public Hitable {
//	Hitable* left;
//	Hitable* right;
//public:
//    __device__ BVH_Node(HitableList* list, curandState* local_rand_state) : BVH_Node(list->list, 0, list->size(), local_rand_state) {}
//    __device__ BVH_Node(Hitable** objects, int start, int end, curandState* local_rand_state) {
//        int axis = ceilf(3*curand_uniform(local_rand_state));
//        auto comparator = (axis == 1) ? box_x_compare
//            : (axis == 2) ? box_y_compare
//            : box_z_compare;
//
//        size_t object_span = end - start;
//
//        if (object_span == 1) {
//            left = right = objects[start];
//        }
//        else if (object_span == 2) {
//            if (comparator(objects[start], objects[start + 1])) {
//                left = objects[start];
//                right = objects[start + 1];
//            }
//            else {
//                left = objects[start + 1];
//                right = objects[start];
//            }
//        }
//        else {
//            sort::quickSort(objects, start, end-1, comparator);
//
//            auto mid = start + object_span / 2;
//            left = new BVH_Node(objects, start, mid, local_rand_state);
//            right = new BVH_Node(objects, mid, end, local_rand_state);
//
//        }
//        aabb = AABB(left->aabb, right->aabb);
//    }
//	__device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
//		if (!aabb.hit(r, tmin, tmax))
//			return false;
//
//		bool hit_left = left->hit(r, tmin, tmax, rec);
//		bool hit_right = right->hit(r, tmin, hit_left ? rec.t : tmax, rec);
//
//		return hit_left || hit_right;
//
//	}
//    __host__ Hitable* toGPU() override {
//        return this;
//    }
//
//private:
//    __device__ static bool box_compare(const Hitable* a, const Hitable* b, int axis_index) {
//        return a->aabb.min[axis_index] < b->aabb.min[axis_index];
//    }
//
//    __device__ static bool box_x_compare(const Hitable* a, const Hitable* b) {
//        return box_compare(a, b, 0);
//    }
//
//    __device__ static bool box_y_compare(const Hitable* a, const Hitable* b) {
//        return box_compare(a, b, 1);
//    }
//
//    __device__ static bool box_z_compare(const Hitable* a, const Hitable* b) {
//        return box_compare(a, b, 2);
//    }
//};