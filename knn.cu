/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "knn.cuh"
#include "gpu_data.h"



__global__
int
cudaGetPairedUserRatingsNumber(float* movie_i, float* movie_j, int size_i, int size_j,) {
    int counter_i = 0;
    int counter_j = 0;
    int res_counter = 0;
    while(counter_i < size_i and counter_j < size_j)
    {
        //
        if ((movie_i[counter_i*3]) == (movie_j[counter_j*3]))
        {
            if ((movie_j[counter_j*3 + 1])!= 0 and 0 != (movie_i[1+3*counter_i]))
            {
                res_counter ++;
            }
            counter_i ++;
            counter_j ++;
        }
        else {
            if ((movie_i[3*counter_i]) > (movie_j[3*counter_j]))
            {
                counter_j ++;
            }
            else {
                counter_i ++;
            }
        }
    }
    return res_counter;
}


__global__
void
cudaGetPairedUserRatings(float* movie_i, float* movie_j, int size_i, int size_j, bool first, float* res) {
    int counter_i = 0;
    int counter_j = 0;
    int res_counter = 0;
    while(counter_i < size_i and counter_j < size_j)
    {
        //
        if ((movie_i[counter_i*3]) == (movie_j[counter_j*3]))
        {
            if ((movie_j[counter_j*3 + 1])!= 0 and 0 != (movie_i[1+3*counter_i]))
            {
                if (first)
                {
                    res[res_counter] = (movie_i[1+3*counter_i]);
                    res_counter ++;
                }
                else{
                    res[res_counter] = (movie_j[1+3*counter_j]);
                    res_counter ++;

                }
            }
            counter_i ++;
            counter_j ++;
        }
        else {
            if ((movie_i[3*counter_i]) > (movie_j[3*counter_j]))
            {
                counter_j ++;
            }
            else {
                counter_i ++;
            }
        }
    }
}


float pearson(float* item_rats_i, float* item_rats_j, int item_rats_size)
{
    float L;
    float top = 0, bottom = 0;
    float item_i_diff[item_rats_size];
    float item_j_diff[item_rats_size];
    float i_sum = 0, j_sum = 0;
    L = item_rats_size;
    if (L <= 1)
    {
        return 0;
    }
    for (int i = 0; i < L; i++)
    {
        i_sum += item_rats_i[i];
        j_sum += item_rats_j[i];
    }
    float x_i_mean = i_sum / L;
    float x_j_mean = j_sum / L;
    float MSE_i = 0;
    float MSE_j = 0;
    
    for(int i = 0; i < L; i++)
    {
        item_i_diff[i] = item_rats_i[i] - x_i_mean;
        item_j_diff[i] = item_rats_j[i] - x_j_mean;
        MSE_i += pow(item_i_diff[i], 2);
        MSE_j += pow(item_j_diff[i], 2);
    }
    for (int i = 0; i < L; i++)
    {
        top += item_i_diff[i]*item_j_diff[i];
    }
    top *= 1/(L-1);
    bottom = sqrt(1/(L-1) * MSE_i)*sqrt(1/(L-1) * MSE_j);
    
    if (bottom == 0)
    {
        return 0;
    }
    return top/bottom;
}

__global__
float get_cij(float* item_rats_i, float* item_rats_j, int alpha, int item_rats_size)
{
    return pearson(item_rats_i, item_rats_j, item_rats_size)*item_rats_size/(item_rats_size+alpha);
}


__global__
void cuda_get_cij_kernel(int user_id, int cur_movie_id, Data *data, float * user_list_movies,
                      float *gpu_out_cij, int cij_length, float **cij_lib, int alpha)
{
    uint thread_index = blockIdx.x * blockDim.x + threadIdx;
    while (thread_index < cij_length) {
        int id = user_list_movies[3*thread_index];
        if (id < cur_movie_id)
        {
            int temp = id; id = cur_movie_id; cur_movie_id = temp;
        }
        if (cij_lib[cur_movie_id][id] == 0)
        {
            float size_i = data->movie_user_num[cur_movie_id];

            float size_j = data->movie_user_num[cur_movie_id];

            
            float* movie_i = data->movie_user[cur_movie_id];
            float* movie_j = data->movie_user[cur_movie_id];
            
            
            int cij_list_size = cudaGetPairedUserRatingsNumber(movie_i, movie_j, size_i, size_j);
            
            float* movie_rat_i;
            float* movie_rat_j;

            cudaMalloc(movie_rat_i, cij_list_size*sizeof(float));
            cudaMalloc(movie_rat_j, cij_list_size*sizeof(float));

            cudaGetPairedUserRatings(movie_i, movie_j, size_i, size_j, true, movie_rat_i);
            cudaGetPairedUserRatings(movie_i, movie_j, size_i, size_j, false, movie_rat_j);
            
            cij_lib[cur_movie_id][id] = get_cij(movie_rat_i, movie_rat_j, alpha);
        }
        gpu_out_cij[id] = cij_lib[cur_movie_id][id];
    }
}


void cudaCallCij(const unsigned int blocks,
        const unsigned int threadsPerBlock, int user_id, int cur_movie_id, Data *data, float * user_list_movies, float *gpu_out_cij, int cij_length, float **cij_lib, int alpha) {
    
    cuda_get_cij_kernel<<<blocks, threadsPerBlock>>>(int user_id, int cur_movie_id, Data *data, float * user_list_movies, float *gpu_out_cij, int cij_length, float **cij_lib, int alpha);
}
