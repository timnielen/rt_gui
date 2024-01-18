#include "sort.h"
#include <iostream>

__device__ void sort::radixSort(unsigned int indices[], unsigned int keys[], unsigned int arrLength, unsigned int keyLength) {
    unsigned int* buckets[2];
    buckets[0] = new unsigned int[arrLength];
    buckets[1] = new unsigned int[arrLength];

    unsigned int bucketSize[2];

    for (int k = 0; k < keyLength; k++) {
        bucketSize[0] = 0;
        bucketSize[1] = 0;
        for (int i = 0; i < arrLength; i++) {
            int index = indices[i];
            int bucket = (1 << k & keys[index]) >> k;
            buckets[bucket][bucketSize[bucket]++] = index;
        }
        for (int i = 0; i < bucketSize[0]; i++)
            indices[i] = buckets[0][i];
        for (int i = 0; i < bucketSize[1]; i++)
            indices[i + bucketSize[0]] = buckets[1][i];
    }

    delete[] buckets[0];
    delete[] buckets[1];
}