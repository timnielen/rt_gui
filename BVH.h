#pragma once
#include <curand_kernel.h>
#include "AABB.h"
#include "Hit.h"
#include "algorithm"
#include "sort.h"
#include "GlobalTypes.h"



class BVH_Node : public Hitable {
public:
    unsigned int left, right;
    bool leftIsLeaf = true;
    bool rightIsLeaf = true;
    __device__ BVH_Node() {}
    __device__ BVH_Node(unsigned int left, unsigned int right, bool leftIsLeaf, bool rightIsLeaf) : left(left), right(right), leftIsLeaf(leftIsLeaf), rightIsLeaf(rightIsLeaf) {}
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override {
        return false;
    }
};

class BVH : public Hitable {
public:
    BVH_Node* nodes = nullptr;
    Hitable** leaves;
    unsigned int countLeaves;

    uint64_t* mortonCodes = nullptr;
    unsigned int* sortedIndices = nullptr;
    unsigned int* leafParents = nullptr;
    unsigned int* nodeParents = nullptr;

    __host__ BVH(Hitable** hlist, int size, AABB aabb);
    void init();

    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override;

    // Get common prefix Length of the morton codes of the sorted leaves at indexA and indexB
    __device__ int prefixLength(unsigned int indexA, unsigned int indexB);
};
__global__ void genAABBs(BVH* bvh, unsigned int* visited);

__global__ void constructBVH(BVH* bvh);

__global__ void copyBvhToHitable(Hitable** hitable, BVH* bvh);

__global__ void printMortonCodes(BVH* bvh);

__global__ void genMortonCodes(BVH* bvh);
