#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


cudaStream_t stream[4];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S, const int offset)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
	
	const int tilesize = 1;

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
 

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	__shared__ float subTileM[16][16];
    __shared__ float subTileN[16][16];

	
	/*
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int m = blockIdx.x;
	int b = blockIdx.z;
	int size = ceil(W/16.0);
	int h = (blockIdx.y/size) * 16 + threadIdx.y;
	int index = blockIdx.y % size;
	int w = (index) * 32 + threadIdx.x;
	*/
	
	
	int b = offset + blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * 16 + ty;
	int col = blockIdx.x * 16 + tx;
	int m = row;
	int h = col/W_out;
	int w = col % W_out;
	
	float currentval = 0;
	
	if(b < B){
		for(int i = 0; i < ceil(C * K * K/16.0); i++){
			int Aindex = i * 16  + ty; int Bindex = i * 16 + tx;
			
			int c = Bindex / (K * K);
			int p =  ((Bindex) % (K * K))/K;
			int q = ((Bindex) % (K * K)) % K;


			if(m < M && Bindex < C * K * K) subTileM[ty][tx] = mask_4d(m, c, p, q);
			else subTileM[ty][tx] = 0;
			
			c = Aindex / (K * K);
			p =  ((Aindex) % (K * K))/K;
			q = ((Aindex) % (K * K)) % K;
			
			if(Aindex < C * K * K && col < H_out * W_out) subTileN[ty][tx] = in_4d(b, c, h * S + p, w * S + q);
			else subTileN[ty][tx] = 0;
			
			__syncthreads();
			
			for(int k = 0; k < 16; k++){
				currentval += subTileM[ty][k] * subTileN[k][tx];
			}
			__syncthreads();
			
		}
	if(m < M && h < H_out && w < W_out) out_4d(b, m, h, w) = currentval;
	}
	
	
	
	
	
	
	/*
	
	float acc = 0.0;
	
	for(int c = 0; c < C; c++){
		for(int p = 0; p < K; p++){
			for(int q = 0; q < K; q++){
				acc = acc + in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
			}
		}
	}
	
	if(h < H_out && w < W_out){
		out_4d(b, m, h, w) = acc;
	}
	
	*/
	



    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__global__ void conv_forward_kernel2(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S, const int offset)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
	
	const int tilesize = 1;

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
 

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	__shared__ float subTileM[32][32];
    __shared__ float subTileN[32][32];

	
	/*
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int m = blockIdx.x;
	int b = blockIdx.z;
	int size = ceil(W/32.0);
	int h = (blockIdx.y/size) * 32 + threadIdx.y;
	int index = blockIdx.y % size;
	int w = (index) * 32 + threadIdx.x;
	*/
	
	
	int b = offset + blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * 32 + ty;
	int col = blockIdx.x * 32 + tx;
	int m = row;
	int h = col/W_out;
	int w = col % W_out;
	
	float currentval = 0;
	
	if(b < B){
		for(int i = 0; i < ceil(C * K * K/32.0); i++){
			int Aindex = i * 32  + ty; int Bindex = i * 32 + tx;
			
			int c = Bindex / (K * K);
			int p =  ((Bindex) % (K * K))/K;
			int q = ((Bindex) % (K * K)) % K;


			if(m < M && Bindex < C * K * K) subTileM[ty][tx] = mask_4d(m, c, p, q);
			else subTileM[ty][tx] = 0;
			
			c = Aindex / (K * K);
			p =  ((Aindex) % (K * K))/K;
			q = ((Aindex) % (K * K)) % K;
			
			if(Aindex < C * K * K && col < H_out * W_out) subTileN[ty][tx] = in_4d(b, c, h * S + p, w * S + q);
			else subTileN[ty][tx] = 0;
			
			__syncthreads();
			
			for(int k = 0; k < 32; k++){
				currentval += subTileM[ty][k] * subTileN[k][tx];
			}
			__syncthreads();
			
		}
	if(m < M && h < H_out && w < W_out) out_4d(b, m, h, w) = currentval;
	}
	
	
	
	
	
	
	/*
	
	float acc = 0.0;
	
	for(int c = 0; c < C; c++){
		for(int p = 0; p < K; p++){
			for(int q = 0; q < K; q++){
				acc = acc + in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
			}
		}
	}
	
	if(h < H_out && w < W_out){
		out_4d(b, m, h, w) = acc;
	}
	
	*/
	



    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
	
	int H_out = (H - K)/S + 1;
    int W_out = (W - K)/S + 1;
	
	int insize = B * C * H * W;
	int outsize = B * M * H_out * W_out;
	int masksize = M * C * K * K;
	
	cudaMalloc((void **) device_input_ptr, insize * sizeof(float));
	cudaMalloc((void **) device_output_ptr, outsize * sizeof(float));
	cudaMalloc((void **) device_mask_ptr, masksize * sizeof(float));
	
	
	cudaMemcpy(*device_mask_ptr, host_mask, masksize * sizeof(float), cudaMemcpyHostToDevice);
	
	int Y = ceil(H_out * W_out/16.0);
	int Y2 = ceil(H_out * W_out/32.0);
	//int Y = ceil(H/32.0) * ceil(W/32.0);
	
	int size = ceil(B/4.0);
	dim3 gridDim(Y, 1, size);
	dim3 blockDim(16, 16, 1);
	
	dim3 gridDim2(Y2, 1, size);
	dim3 blockDim2(32, 32, 1);
	
	for(unsigned int i =0; i < 4; i++){
		cudaStreamCreate(&stream[i]);
		cudaMemcpyAsync(&(*device_input_ptr)[i * size * C * H * W], &host_input[i * size * C * H * W], insize/B * sizeof(float) * size, cudaMemcpyHostToDevice, stream[i]);
		if(H_out % 32 == 0) conv_forward_kernel2<<<gridDim2, blockDim2, 0, stream[i]>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, B, M, C, H, W, K, S, i * size);
		else conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, B, M, C, H, W, K, S, i * size);
	}
	
	
	
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

	/*
	const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
	
    int Y = ceil(H_out * W_out/16.0);
	
	dim3 gridDim(Y, 1, B);
	dim3 blockDim(16, 16, 1);
	

	conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
	*/
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
	
	int H_out = (H - K)/S + 1;
    int W_out = (W - K)/S + 1;
	
	int outsize = B * M * H_out * W_out;
	
	int size = ceil(B/4.0);
	for(unsigned int i =0; i < 4; i++){
		cudaMemcpyAsync(&host_output[i * size * M * H_out * W_out], &device_output[i * size * M * H_out * W_out], outsize/B * sizeof(float) * size, cudaMemcpyDeviceToHost, stream[i]);
	}
	
	//cudaMemcpy(host_output, device_output, outsize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_output);
	cudaFree(device_input);
	cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
