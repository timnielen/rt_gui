#include "sort.h"
#include <iostream>

__device__ void sort::radixSort(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength) {
    unsigned int* buckets[2];
    buckets[0] = new unsigned int[arrLength];
    buckets[1] = new unsigned int[arrLength];

    unsigned int bucketSize[2];

    for (int k = 0; k < keyLength; k++) {
        bucketSize[0] = 0;
        bucketSize[1] = 0;
        for (int i = 0; i < arrLength; i++) {
            int index = indices[i];
            int bucket = (1ull << k & keys[index]) >> k;
            buckets[bucket][bucketSize[bucket]++] = index;
        }
        //printf("key: %d, buckets[0][0]: %d, buckets[1][0]: %d\n", k, buckets[0][0], buckets[1][0]);
        /*for (int i = 0; i < bucketSize[0]; i++)
            indices[i] = buckets[0][i];
        for (int i = 0; i < bucketSize[1]; i++)
            indices[i + bucketSize[0]] = buckets[1][i];*/
        for (int i = 0; i < arrLength; i++) {
            if (i < bucketSize[0])
                indices[i] = buckets[0][i];
            else
                indices[i] = buckets[1][i - bucketSize[0]];
        }
    }

    delete[] buckets[0];
    delete[] buckets[1];
}

__global__ void parallelRadixSortKernel(unsigned int indices[], uint64_t keys[], const unsigned int arrLength, const unsigned int keyLength, const unsigned int parts, const unsigned int partSize, unsigned int tmpIndices[]) {
    extern __shared__ unsigned int cnt[];
    const int part = threadIdx.x;
    const int start = part * partSize;
    if (start >= arrLength)
        return;
    const int bucketCount = 2;
    const int end = fminf(start + partSize, arrLength);
    for (int key = 0; key < keyLength; key++) {
        cnt[bucketCount * part] = 0;
        cnt[bucketCount * part + 1] = 0;
        for (int i = start; i < end; i++) {
            int index = indices[i];
            int bucket = ((1ull << key) & keys[indices[i]]) >> key;
            cnt[bucketCount * part + bucket]++;
        }
        __syncthreads();
        if (part == 0)
        {
            unsigned int offset = 0;
            for (int bucket = 0; bucket < bucketCount; bucket++) {
                for (int p = 0; p < parts; p++) {
                    offset += cnt[bucketCount * p + bucket];
                    cnt[bucketCount * p + bucket] = offset - cnt[bucketCount * p + bucket];
                }
            }
        }
        __syncthreads();
        for (int i = start; i < end; i++) {
            tmpIndices[i] = indices[i];
        }
        __syncthreads();
        for (int i = start; i < end; i++) {
            unsigned int index = tmpIndices[i];
            int bucket = ((1ull << key) & keys[index]) >> key;
            indices[cnt[bucketCount * part + bucket]++] = index;
        }
        __syncthreads();
    }
}

void sort::parallelRadixSort(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength) {
    const int blockSize = 1024;
    const int partSize = (arrLength + blockSize - 1) / blockSize;
    const int parts = (arrLength + partSize - 1) / partSize;
    unsigned int* tmpIndices;
    cudaMalloc((void**)&tmpIndices, sizeof(unsigned int) * arrLength);
    parallelRadixSortKernel << <1, blockSize, 2 * parts * sizeof(unsigned int) >> > (indices, keys, arrLength, keyLength, parts, partSize, tmpIndices);
    cudaFree(tmpIndices);
}