/* CUDA blur
 * Kevin Yuh, 2014 */

#ifndef CUDA_FFT_CONVOLVE_CUH
#define CUDA_FFT_CONVOLVE_CUH



#include <cufft.h>

float correlationKernelSum(float* cij, int total_size, int);

int cudaGetPairedUserRatingsNumber(float*, int size_i, int, int );

void merge(float *src, float *dst, float* followsrc, float* followdst, int low, int mid, int hi);

void callMergeKernel(const unsigned int blocks, const unsigned intthreadsPerBlock, float*, float*, int);

 void
cudaGetPairedUserRatings(float* movie_i, float* movie_j, int size_i, bool first, float* res);

 float pearson(float* item_rats_i, float* item_rats_j, int item_rats_size, float *, float *); 

float get_cij(float* item_rats_i, float* item_rats_j, int alpha, int item_rats_size, float *, float *);


void cudaCallCij(const unsigned int blocks,
        const unsigned int threadsPerBlock, int user_id, int cur_movie_id, int, float*, float * user_list_movies, float *gpu_out_cij, int cij_length, float *cij_lib, int alpha);
#endif
