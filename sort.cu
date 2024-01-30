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

#define BUCKET_COUNT 2
__global__ void parallelRadixSortKernel(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength, unsigned int parts, const unsigned int partSize) {
    ////fist buckets = buckets[0], buckets[1] concatenated
    extern __shared__ uint32_t cnt[];
    //uint32_t cnt[parts][BUCKET_COUNT];
    __shared__ unsigned int* shPointer[1];
    int part = threadIdx.x;
    int start = part * partSize;
    if (start >= arrLength) return;
    int end = fminf(start + partSize, arrLength);
    if (part == 0)
        printf("arrLength: %d\n", arrLength);
    if (part == 0) {
        *shPointer = new unsigned int[arrLength];
    }
    __syncthreads();
    unsigned int* tmpIndices = *shPointer;
    for (int key = 0; key < keyLength; key++) {
        cnt[BUCKET_COUNT * part] = 0;
        cnt[BUCKET_COUNT * part + 1] = 0;
        __syncthreads();
        for (int i = start; i < end; i++) {
            int index = indices[i];
            int bucket = ((1ull << key) & keys[indices[i]]) >> key;
            cnt[BUCKET_COUNT * part + bucket]++;
        }
        __syncthreads();
        if (part == 0)
        {
            unsigned int offset = 0;
            unsigned int tmp = 0;
            for (int bucket = 0; bucket < BUCKET_COUNT; bucket++) {
                for (int part = 0; part < parts; part++) {
                    tmp = cnt[BUCKET_COUNT * part + bucket];
                    cnt[BUCKET_COUNT * part + bucket] = offset;
                    offset += tmp;
                }
            }
        }
        __syncthreads();
        for (int i = start; i < end; i++) {
            tmpIndices[i] = indices[i];
        }
        for (int i = start; i < end; i++) {
            unsigned int index = tmpIndices[i];
            int bucket = ((1ull << key) & keys[tmpIndices[i]]) >> key;
            unsigned int el = cnt[BUCKET_COUNT * part + bucket]++;
            indices[el] = index;
        }
        __syncthreads();
    }
    //delete[] tmpIndices;
}

#define BLOCK_SIZE 16
void sort::parallelRadixSort(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength) {
    const int partSize = (arrLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parallelRadixSortKernel << <1, BLOCK_SIZE, BLOCK_SIZE* BUCKET_COUNT * sizeof(unsigned int) >> > (indices, keys, arrLength, keyLength, BLOCK_SIZE, partSize);
}