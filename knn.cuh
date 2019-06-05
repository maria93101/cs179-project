/* CUDA blur
 * Kevin Yuh, 2014 */

#ifndef CUDA_FFT_CONVOLVE_CUH
#define CUDA_FFT_CONVOLVE_CUH



#include <cufft.h>

int cudaGetPairedUserRatingsNumber(float* movie_i, float* movie_j, int size_i, int size_j);

void
cudaGetPairedUserRatings(float* movie_i, float* movie_j, int size_i, int size_j, bool first, float* res);

float pearson(float* item_rats_i, float* item_rats_j, int item_rats_size); 

float get_cij(float* item_rats_i, float* item_rats_j, int alpha, int item_rats_size);

void cuda_get_cij_kernel(int user_id, int cur_movie_id, Data *data, float * user_list_movies,
                      float *gpu_out_cij, int cij_length, float **cij_lib, int alpha);

void cudaCallCij(const unsigned int blocks,
        const unsigned int threadsPerBlock, int user_id, int cur_movie_id, Data *data, float * user_list_movies, float *gpu_out_cij, int cij_length, float **cij_lib, int alpha);
#endif
