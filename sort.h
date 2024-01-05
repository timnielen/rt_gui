#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

namespace sort {
    template<class T>
    __device__ inline void swap(T& a, T& b) {
        T temp = a;
        a = b;
        b = temp;
    }


    template<class T, typename OP>
    __device__ int partition(T arr[], int start, int end, OP compare)
    {

        T pivot = arr[start];

        int count = 0;
        for (int i = start + 1; i <= end; i++) {
            if (compare(arr[i], pivot))
                count++;
        }

        // Giving pivot element its correct position
        int pivotIndex = start + count;
        swap(arr[pivotIndex], arr[start]);

        // Sorting left and right parts of the pivot element
        int i = start, j = end;

        while (i < pivotIndex && j > pivotIndex) {

            while (compare(arr[i], pivot)) {
                i++;
            }

            while (!compare(arr[j], pivot)) {
                j--;
            }

            if (i < pivotIndex && j > pivotIndex) {
                swap(arr[i++], arr[j--]);
            }
        }

        return pivotIndex;
    }

    template<class T, typename OP>
    __device__ void quickSort(T arr[], int start, int end, OP compare)
    {

        // base case
        if (start >= end)
            return;

        // partitioning the array
        int p = partition(arr, start, end, compare);

        // sorting the left part
        quickSort(arr, start, p - 1, compare);

        // sorting the right part
        quickSort(arr, p + 1, end, compare);
    }
}
