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
		nodes = (countLeaves == 1) ? nullptr : new BVH_Node[countLeaves - 1];
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

		printf("index,ptr,leftptr,rightptr,leftLeaf,rightLeaf\n");
        for (int i = 0; i < countLeaves - 1; i++)
            printf("%d,%p,%p,%p,%d,%d\n", i, nodes + i, nodes[i].left, nodes[i].right, nodes[i].leftIsLeaf, nodes[i].rightIsLeaf);
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