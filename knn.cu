/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "knn.cuh"
//#include "gpu_data.h"
__device__ 
void merge(float *src, float *dst, float* followsrc, float* followdst, int low, int mid, int hi)
{
	a_counter = low;
	b_counter = mid;
	for (int i = low; i < hi; i++) {
		if (a_counter < middle && (b_counter >= end || source[a_counter] < source[b_counter])) {
			dst[i] = src[a_counter];
			followdst[i] = followsrc[a_counter];
			a_counter ++;
		}
		else
		{
			dst[i] = src[b_counter];
			followdst[i] = followsrc[b_counter];
			b_counter ++;
		}
	}
}

__global__
void mergeSortKernel( float *src, float *dst, float * followsrc, float * followdst, int section, int num_section, int total_size)
{
	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int low = thread_index * section * num_section;
	int mid, hi; 	
	int slice = 0;
	// Now we do stuff for each slice. 
	while(low < total_size)
	{
		mid = min(low + section/2, total_size);
		hi = min(low + section, total_size);
		merge(src, des, followsrc, followdst,low, mid, hi);
		low += section;
		slice ++;
	}
	
}


void callMergeKernel(blocks, threadsPerBlock, cij, cij_dest, cijr, cijr_dest, section, num_section, total_size)
{
	for (int section = 2; section< total_size /2; section <<= 1) {
		int num_section = size / ((total_threads) * section) + 1;
		gpu_mergesort<<<blocks, threadsPerBlock>>>(cij, cij_dest, cijr, cijr_dest, section, num_section, total_size);
		int *temp = cij_dest;
		cij_dest = cij;
		cij = temp;

		temp = cijr_dest;
		cijr_dest = cijr;
		cijr = temp;
	
	}
}

__device__
int
cudaGetPairedUserRatingsNumber(float* data_lib, int size_i, int cij_length, int tid) {
    int counter_i = 0;
    int counter_j = 0;
    int res_counter = 0;
    while(counter_i < size_i and counter_j < size_i)
    {
        printf("id: %d jid: %d \n", data_lib[tid*size_i + counter_i*3], data_lib[cij_length*size_i + counter_j*3]);
		printf("counter i: %d, counter j: %d \n", counter_i, counter_j);
		if (data_lib[tid*size_i + counter_i*3] <= 0)
		{
			counter_i ++;
		}
		else if(data_lib[cij_length*size_i + counter_j*3] <= 0)
		{
			counter_j ++;
		}
        else if ((data_lib[tid*size_i + counter_i*3]) == (data_lib[cij_length*size_i + counter_j*3]))
        {
            if ((data_lib[cij_length*size_i + counter_j*3 + 1])!= 0 and 0 != (data_lib[tid*size_i + 1+3*counter_i]))
            {
                res_counter ++;
            }
            counter_i ++;
            counter_j ++;
        }
        else {
            if ((data_lib[tid*size_i + 3*counter_i]) > (data_lib[cij_length*size_i + 3*counter_j]))
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


__device__ void
cudaGetPairedUserRatings(float* movie_i, float* movie_j, int size_i, bool first, float* res) {
    int counter_i = 0;
    int counter_j = 0;
    int res_counter = 0;
    while(counter_i < size_i and counter_j < size_i)
    {
        //
        if (movie_i[counter_i*3] < 0)
        {   
            counter_i ++; 
        }   
        else if(movie_j[counter_j*3] < 0)
        {   
            counter_j ++; 
        }   
        else if ((movie_i[counter_i*3]) == (movie_j[counter_j*3]))
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
__device__
float pearson(float* item_rats_i, float* item_rats_j, int item_rats_size, float *item_i_diff, float *item_j_diff)
{
    float L;
    float top = 0, bottom = 0;
	//cudaMalloc((void **) &item_j_diff, item_rats_size*sizeof(float));
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
	delete [] item_j_diff;
	delete [] item_i_diff;
	//cudaFree(item_i_diff);
	//cudaFree(item_j_diff);
    return top/bottom;
}

__device__
float get_cij(float* item_rats_i, float* item_rats_j, int alpha, int item_rats_size, float *item_diff_i, float * item_diff_j)
{
    return pearson(item_rats_i, item_rats_j, item_rats_size, item_diff_i, item_diff_j)*item_rats_size/(item_rats_size+alpha);
}


__global__
void cuda_get_cij_kernel(int user_id, int cur_movie_id, int gpu_data_max_size, float * gpu_data_lib, float * user_list_movies,
                      float *gpu_out_cij, int cij_length, float *cij_lib, int alpha)
{
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < cij_length) {
	    int id = user_list_movies[3*thread_index];
        if (id < cur_movie_id)
        {
            int temp = id; id = cur_movie_id; cur_movie_id = temp;
        }
        if (cij_lib[cur_movie_id*17770 + id] <= 0)
        {
            //float* movie_i = (float*)malloc(gpu_data_max_size * sizeof(float));
			//float* movie_j = (float*)malloc(gpu_data_max_size * sizeof(float));

			//float *item_j_diff = (float * )malloc(gpu_data_max_size * sizeof(float));
			//float *item_i_diff = (float*)malloc(gpu_data_max_size * sizeof(float));

			//if (movie_i == NULL) {printf(" rip m i");} else {printf("yay moive_i ");}
			//if (movie_j == NULL) {printf(" rip movie j ");}else {printf("yay moive_j ");}
			//if (item_i_diff == NULL) {printf(" rip item_i_diff ");}else {printf("yay item_i ");}
			//if (item_j_diff == NULL) {printf(" rip item_J_diff ");}else {printf("yay itme_j ");}
			//for (int s = 0; s < 1; s++)
			//{
			//	printf("s: %d, gpu_data_max_size: %d, thread_index: %d, indexing:  %d \n", s, gpu_data_max_size, thread_index, cij_length * gpu_data_max_size + s);	
			//	movie_i[s] = gpu_data_lib[cij_length * gpu_data_max_size + s];
				//movie_j[s] = gpu_data_lib[thread_index * gpu_data_max_size + s];
			//}
			
			//memcpy((void *) &movie_i,  (void *) &gpu_data_lib[cij_length * gpu_data_max_size], gpu_data_max_size * sizeof(float));
			//memcpy((void *) &movie_j,  (void *) &gpu_data_lib[thread_index * gpu_data_max_size], gpu_data_max_size * sizeof(float));

            int cij_list_size = cudaGetPairedUserRatingsNumber(gpu_data_lib, gpu_data_max_size, cij_length, thread_index);
            
            //float* movie_rat_i = new float[cij_list_size];
            //float* movie_rat_j = new float[cij_list_size];

            //movie_rat_i = malloc(cij_list_size*sizeof(float));
            //movie_rat_j = malloc(cij_list_size*sizeof(float));

            //cudaGetPairedUserRatings(movie_i, movie_j, gpu_data_max_size, true, movie_rat_i);
            //cudaGetPairedUserRatings(movie_i, movie_j, gpu_data_max_size, false, movie_rat_j);
            cij_lib[cur_movie_id*17770 + id] = cij_list_size;//get_cij(movie_rat_i, movie_rat_j, alpha, cij_list_size, item_i_diff, item_j_diff);
			//delete [] movie_rat_i;
			//delete [] movie_rat_j;

			//delete [] movie_i;
			//free(movie_i);
//			free(movie_j);
			//free(item_j_diff);
			//free(item_i_diff);
        }
        gpu_out_cij[thread_index] = 1;//cij_lib[cur_movie_id*17770 +id];
		thread_index += blockDim.x * gridDim.x;
    }
}

void cudaCallCij(const unsigned int blocks,
        const unsigned int threadsPerBlock, int user_id, int cur_movie_id, int gpu_data_max_size, float * gpu_data_lib, float * user_list_movies, float *gpu_out_cij, int cij_length, float *cij_lib, int alpha) {
    
    cuda_get_cij_kernel<<<blocks, threadsPerBlock>>>(user_id, cur_movie_id, gpu_data_max_size, gpu_data_lib, user_list_movies, gpu_out_cij, cij_length, cij_lib, alpha);
}
