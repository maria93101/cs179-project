/* CUDA blur
 * Kevin Yuh, 2014 */

#ifndef CUDA_FFT_CONVOLVE_CUH
#define CUDA_FFT_CONVOLVE_CUH



#include <cufft.h>

float correlationKernelSum(float* cij, int total_size, int);

void merge(float *src, float *dst, float* followsrc, float* followdst, int low, int mid, int hi);

void callMergeKernel(const unsigned int blocks, const unsigned intthreadsPerBlock, float*, float*, int);
#endif
