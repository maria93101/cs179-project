
#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "knn.cuh"
//#include "gpu_data.h"
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }   
}

#define BW 1024

/*
 * This function merges 2 lists [low to mid), [mid to hi) not in places.
*/
__device__ 
void merge(float *src, float *dst, float* followsrc, float* followdst, int low, int mid, int hi)
{
	int a_counter = low;
	int b_counter = mid;
	for (int i = low; i < hi; i++) {
		if (a_counter < mid && (b_counter >= hi || src[a_counter] > src[b_counter])) {
			dst[i] = src[a_counter];
			followdst[i] = followsrc[a_counter];
			a_counter ++;
		}
		else
		{
			if (src[a_counter] == src[b_counter])
			{
				if (followsrc[a_counter] > followsrc[b_counter])
				{
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
			else
			{
				dst[i] = src[b_counter];
				followdst[i] = followsrc[b_counter];
				b_counter ++;
			}
		}
	}
}

__global__ void correlationKernel(float* cij, float*sum, int total_size, int begin)
{
    extern __shared__ float shmem[];

     // atomically add the accumulated loss per block into the global accumulator
	uint s_thread_index = threadIdx.x;
	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	while(thread_index < total_size)
	{
		shmem[s_thread_index] = cij[thread_index];

		thread_index += blockDim.x * gridDim.x;
	}	
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
    	if(s_thread_index < stride)
        {
         	shmem[s_thread_index] += shmem[s_thread_index + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicAdd(sum, shmem[0]);
    }
}

float correlationKernelSum(float* cij, int total_size, int begin)
{
    // Inialize loss on the device to be zero
    float sum, *d_sum;
    gpu_errchk( cudaMalloc(&d_sum, sizeof(float)) );
    cudaMemsetType<float>(d_sum, 0.0, 1);

    float *gpu_cij;
    gpu_errchk( cudaMalloc(&gpu_cij, total_size * sizeof(float)) );
    gpu_errchk(cudaMemcpy(gpu_cij, cij, total_size * sizeof(float),
                cudaMemcpyHostToDevice));

    // Accumulate the total loss on the device by invoking a kernel
    int n_blocks = std::min(65535, (total_size + BW  - 1) / BW);

    correlationKernel <<<n_blocks,  BW, BW * sizeof(float)>>>(gpu_cij, d_sum, total_size, begin);

    gpu_errchk( cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost) );
    gpu_errchk( cudaFree(d_sum) );
	gpu_errchk( cudaFree(gpu_cij) );
    // Return the sum
    return sum;
}


__global__
void mergeSortKernel( float *src, float*dst, float * followsrc, float * followdst, int section, int num_section, int total_size)
{
	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int low = thread_index * section * num_section;
	int mid, hi; 	
	int slice = 0;

	//printf("threadidx: %s", thread_index);
	// Now we do stuff for each slice. 
	while(slice < num_section && low < total_size)
	{
		mid = min(low + section/2, total_size);
		hi = min(low + section, total_size);
		merge(src, dst, followsrc, followdst,low, mid, hi);
		low += section;
		slice ++;
	}
	
}


void callMergeKernel(const unsigned int blocks, const unsigned int threadsPerBlock, float * cij, float * cijr, int total_size)
{
    //Allocate GPU...
    float *gpu_src;
    gpu_errchk(cudaMalloc((void **) &gpu_src, total_size * sizeof(float))); 
    gpu_errchk(cudaMemcpy(gpu_src, cij, total_size * sizeof(float),
                   cudaMemcpyHostToDevice));

    float *gpu_dst;
    gpu_errchk(cudaMalloc((void **) &gpu_dst, total_size * sizeof(float)));
    gpu_errchk(cudaMemset(gpu_dst, 0, total_size * sizeof(float)));

    float *gpu_fsrc;
    gpu_errchk(cudaMalloc((void **) &gpu_fsrc, total_size * sizeof(float)));
    gpu_errchk(cudaMemcpy(gpu_fsrc, cijr, total_size * sizeof(float),
                cudaMemcpyHostToDevice));

    float *gpu_fdst;
    gpu_errchk(cudaMalloc((void **) &gpu_fdst, total_size * sizeof(float)));
    gpu_errchk(cudaMemset(gpu_fdst, 0, total_size * sizeof(float)));

	int total_threads = blocks * threadsPerBlock;
	for (int section = 2; section< total_size *2; section <<= 1) {
		int num_section = total_size / ((total_threads) * section) + 1;
		mergeSortKernel<<<blocks, threadsPerBlock>>>(gpu_src, gpu_dst, gpu_fsrc, gpu_fdst, section, num_section, total_size);
		float *temp = gpu_dst;
		gpu_dst = gpu_src;
		gpu_src = temp;

		temp = gpu_fdst;
		gpu_fdst = gpu_fsrc;
		gpu_fsrc = temp;
	
	}

	gpu_errchk(cudaMemcpy(cij, gpu_src, total_size * sizeof(float),
                   cudaMemcpyDeviceToHost));

    gpu_errchk(cudaMemcpy(cijr, gpu_fsrc, total_size * sizeof(float),
                   cudaMemcpyDeviceToHost));
	gpu_errchk( cudaFree(gpu_fsrc) );
	gpu_errchk( cudaFree(gpu_src) );
	gpu_errchk( cudaFree(gpu_dst) );
	gpu_errchk( cudaFree(gpu_fdst) );	
}


