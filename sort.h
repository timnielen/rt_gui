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
    __device__ int partition(T arr[], int l, int h, OP compare)
    {
        T x = arr[h];
        int i = (l - 1);

        for (int j = l; j <= h - 1; j++) {
            if (compare(arr[j], x)) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[h]);
        return (i + 1);
    }

    /* A[] --> Array to be sorted,
    l --> Starting index,
    h --> Ending index */
    template<class T, typename OP>
    __device__ void quickSort(T arr[], int l, int h, OP compare)
    {
        // Create an auxiliary stack 
        int *stack = new int[h - l + 1];

        // initialize top of stack 
        int top = -1;

        // push initial values of l and h to stack 
        stack[++top] = l;
        stack[++top] = h;

        // Keep popping from stack while is not empty 
        while (top >= 0) {
            // Pop h and l 
            h = stack[top--];
            l = stack[top--];

            // Set pivot element at its correct position 
            // in sorted array 
            int p = partition(arr, l, h, compare);

            // If there are elements on left side of pivot, 
            // then push left side to stack 
            if (p - 1 > l) {
                stack[++top] = l;
                stack[++top] = p - 1;
            }

            // If there are elements on right side of pivot, 
            // then push right side to stack 
            if (p + 1 < h) {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
        delete[] stack;
    }
}