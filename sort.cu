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
        printf("key: %d, buckets[0][0]: %d, buckets[1][0]: %d\n", k, buckets[0][0], buckets[1][0]);
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
__global__ void parallelRadixSortKernel(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength, unsigned int parts, unsigned int partSize) {
    //fist buckets = buckets[0], buckets[1] concatenated
    extern __shared__ uint32_t cnt[];
    //uint32_t cnt[parts][BUCKET_COUNT];
    int part = threadIdx.x;
    int start = part * partSize;
    if (start >= arrLength) return;
    int end = fminf(start + partSize, arrLength);

    unsigned int* tmpIndices = new unsigned int[end - start];
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
            uint32_t offset = 0;
            for (int bucket = 0; bucket < BUCKET_COUNT; bucket++) {
                for (int part = 0; part < parts; part++) {
                    uint32_t tmp = cnt[BUCKET_COUNT * part + bucket];
                    cnt[BUCKET_COUNT * part + bucket] = offset;
                    offset += tmp;
                    
                }
            }
        }
        __syncthreads();
        printf("key: %d, bucket: %d, part: %d, %d\n", key, 0, part, cnt[BUCKET_COUNT * part + 0]);
        printf("key: %d, bucket: %d, part: %d, %d\n", key, 1, part, cnt[BUCKET_COUNT * part + 1]);
        for (int i = start; i < end; i++) {
            tmpIndices[i-start] = indices[i];
        }
        for (int i = start; i < end; i++) {
            unsigned int index = tmpIndices[i-start];
            int bucket = ((1ull << key) & keys[indices[i]]) >> key;
            uint32_t el = cnt[BUCKET_COUNT * part + bucket]++;
            if (el < arrLength)
                indices[el] = index;
            /*else
                printf("el: %d, part: %d, bucket %d\n", el, part, bucket);*/
        }
        __syncthreads();
    }
    delete[] tmpIndices;
}

#define BLOCK_SIZE 32
__device__ void sort::parallelRadixSort(unsigned int indices[], uint64_t keys[], unsigned int arrLength, unsigned int keyLength) {
    const uint32_t parts = BLOCK_SIZE;
    const int partSize = (arrLength + parts - 1) / parts;
    parallelRadixSortKernel << <1, parts, BUCKET_COUNT * parts * sizeof(uint32_t) >> > (indices, keys, arrLength, keyLength, parts, partSize);
}