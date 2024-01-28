#include "BVH.h"
#include "cuda_runtime.h"
#include "Material.h"
#include "stdint.h"
#include <inttypes.h>

#define MORTON_LENGTH 63

//__device__ unsigned int expandBits(unsigned int v)
//{
//	v = (v * 0x00010001u) & 0xFF0000FFu;
//	v = (v * 0x00000101u) & 0x0F00F00Fu;
//	v = (v * 0x00000011u) & 0xC30C30C3u;
//	v = (v * 0x00000005u) & 0x49249249u;
//	return v;
//}
//
//// Calculates a 30-bit Morton code for the
//// given 3D point located within the unit cube [0,1].
//__device__ unsigned int morton3D(Vec3 v)
//{
//	float x = fmin(fmax(v.x * 1024.0f, 0.0f), 1023.0f);
//	float y = fmin(fmax(v.y * 1024.0f, 0.0f), 1023.0f);
//	float z = fmin(fmax(v.z * 1024.0f, 0.0f), 1023.0f);
//	unsigned int xx = expandBits((unsigned int)x);
//	unsigned int yy = expandBits((unsigned int)y);
//	unsigned int zz = expandBits((unsigned int)z);
//	return xx * 4 + yy * 2 + zz;
//}

__device__ uint64_t splitBy3(unsigned int a) {
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}


// Calculates a 63-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ uint64_t morton3D(Vec3 v) {
	unsigned int x = (unsigned int)fmin(fmax(v.x * 2097152.0f, 0.0f), 2097151.0f);
	unsigned int y = (unsigned int)fmin(fmax(v.y * 2097152.0f, 0.0f), 2097151.0f);
	unsigned int z = (unsigned int)fmin(fmax(v.z * 2097152.0f, 0.0f), 2097151.0f);
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}


__device__ int BVH::prefixLength(unsigned int indexA, unsigned int indexB) {
	if (indexB < 0 || indexB > countLeaves - 1) return -1;
	uint64_t keyA = mortonCodes[sortedIndices[indexA]];
	uint64_t keyB = mortonCodes[sortedIndices[indexB]];
	if (keyA == keyB)
		return MORTON_LENGTH + __clz(indexA ^ indexB);
	return __clzll(keyA ^ keyB);
}

__device__ void BVH::genMortonCodes() {
	mortonCodes = new uint64_t[countLeaves];
	sortedIndices = new unsigned int[countLeaves];
	Vec3 dimensions = aabb.max - aabb.min;
	for (unsigned int i = 0; i < countLeaves; i++) {
		sortedIndices[i] = i;
		AABB child = leaves[i]->aabb;
		Vec3 centroid = (child.min + child.max) / 2;
		centroid = centroid - aabb.min;
		centroid /= dimensions;
		mortonCodes[i] = morton3D(centroid);
	}

	sort::radixSort(sortedIndices, mortonCodes, countLeaves, MORTON_LENGTH);
	/*for (int i = 0; i < countLeaves; i++) {
		printf("%016x\n", mortonCodes[sortedIndices[i]]);
	}*/
}


// implementation of Karras et al. 2012
__global__ 
void constructBVH(BVH* bvh) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > bvh->countLeaves - 2 || bvh->nodes == nullptr) return;

	//Determine direction of the range (+1 or -1)
	int direction = ((bvh->prefixLength(index, index + 1) - bvh->prefixLength(index, index - 1)) < 0) ? -1 : 1;

	//Compute upper bound for the length of the range
	int deltaMin = bvh->prefixLength(index, index - direction);
	int lenMax = 2;

	while (bvh->prefixLength(index, index + lenMax * direction) > deltaMin)
		lenMax *= 2;

	//Find the other end using binary search
	int len = 0;

	for (int t = lenMax / 2; t >= 1; t /= 2) {
		if (bvh->prefixLength(index, index + (len + t) * direction) > deltaMin) {
			len += t;
		}
	}
	int indexEnd = index + len * direction;

	// Find the split position using binary search
	int deltaNode = bvh->prefixLength(index, indexEnd);
	int split = 0;

	for (int i = 2; ((len + i - 1) / i) >= 1; i *= 2) {
		int t = ((len + i - 1) / i);
		if (bvh->prefixLength(index, index + (split + t) * direction) > deltaNode)
			split += t;
	}

	int splitPos = index + (split) * direction + min(direction, 0);

	bool leftLeaf = min(index, indexEnd) == splitPos;
	bool rightLeaf = max(index, indexEnd) == splitPos + 1;
	bvh->nodes[index] = BVH_Node(splitPos, leftLeaf, rightLeaf);
	AABB aabb = bvh->leaves[bvh->sortedIndices[index]]->aabb;
	for (int i = 1; i <= len; i++)
		aabb = AABB(aabb, bvh->leaves[bvh->sortedIndices[index + i * direction]]->aabb);
	bvh->nodes[index].aabb = aabb;
}


__device__ bool BVH::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const  {
	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	if (!aabb.hit(r, tmin, tmax))
		return false;
	if (nodes == nullptr)
		return leaves[0]->hit(r, tmin, tmax, rec);
	BVH_Node* stack[128];
	BVH_Node** stackPtr = stack;
	*stackPtr++ = NULL; // push

	//// Traverse nodes starting from the root.
	HitRecord temp_rec;
	bool hit_anything = false;
	float closest_so_far = tmax;
	BVH_Node* node = nodes;
	do
	{
		// Check each child node for overlap.
		Hitable* childL = (node->leftIsLeaf) ? leaves[sortedIndices[node->splitPos]] : &nodes[node->splitPos];
		Hitable* childR = (node->rightIsLeaf) ? leaves[sortedIndices[node->splitPos+1]] : &nodes[node->splitPos+1];
		bool overlapL = childL->aabb.hit(r, tmin, closest_so_far);
		bool overlapR = childR->aabb.hit(r, tmin, closest_so_far);

		// Query overlaps a leaf node => report collision.

		if (overlapL && node->leftIsLeaf)
		{
			if (childL->hit(r, tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		if (overlapR && node->rightIsLeaf)
		{
			if (childR->hit(r, tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !node->leftIsLeaf);
		bool traverseR = (overlapR && !node->rightIsLeaf);

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? (BVH_Node*)childL : (BVH_Node*)childR;
			if (traverseL && traverseR)
				*stackPtr++ = (BVH_Node*)childR; // push
		}
	} while (node != NULL);
	//delete[] stack;
	return hit_anything;
}

//__device__ bool BVH::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {
//	if (!aabb.hit(r, tmin, tmax))
//		return false;
//	HitRecord temp_rec;
//	bool hit_anything = false;
//	float closest_so_far = tmax;
//	for (unsigned int i = 0; i < countLeaves; i++) {
//		if (leaves[i]->hit(r, tmin, closest_so_far, temp_rec)) {
//			hit_anything = true;
//			closest_so_far = temp_rec.t;
//			rec = temp_rec;
//		}
//	}
//	return hit_anything;
//}
