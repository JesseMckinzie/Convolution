#include "convolution.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include <string.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

//CUFFT Header file
#include <cufftXt.h>

// helper functions and utilities to work with CUDA

#define BLOCK_SIZE 32 // BLOCK_SIZE*BLOCK_SIZE = max number of threads per block
#define BATCH 10

typedef float2 Complex;

// Forward Declaration

__global__ void solvePoisson(cufftComplex *, cufftComplex *, float *, int, int, int);

using namespace std;

__global__ void convolve2d_helper(float* image, int row_size, int col_size, float* kernel, int kernel_size, int kernel_offset, float* result){

    // calculate row and column positions
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // check bounds
    if(row >= row_size || col >= col_size) return;

    int iFlip, jFlip; // flipped kernel indices
    int ii, jj;
    float temp = 0;

    for(int i = 0; i < kernel_size; ++i){

        iFlip = kernel_size - 1 - i;

        for(int j = 0; j < kernel_size; ++j){

            jFlip = kernel_size - 1 - j;

            ii = row + (kernel_offset - iFlip);
            jj = col + (kernel_offset - jFlip);

            if(ii >= 0 && ii < row_size && jj >= 0 && jj < col_size) {
                temp += image[ii * col_size + jj] * kernel[iFlip * kernel_size + jFlip];
            }
        }
    }

    result[row * col_size + col] = temp;
}


__global__ void convolve2d_helper_opt(float* image, int row_size, int col_size, const float* __restrict__ kernel, int kernel_size, int kernel_offset, float* result, int tile_width){
    
    // calculate row and column positions
    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    int m_row = row - kernel_offset;
    int m_col = col - kernel_offset;

    __shared__ float tile[BLOCK_SIZE * BLOCK_SIZE];

    if (m_row >= 0 && m_row < row_size && m_col >=0 && m_col < col_size) {
        tile[threadIdx.y * BLOCK_SIZE + threadIdx.x] = image[m_row * col_size + m_col];
    } else {
        tile[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
    }

    __syncthreads();

    float temp = 0;
    if (threadIdx.y < tile_width && threadIdx.x < tile_width && row < row_size  && col < col_size) {

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                temp += kernel[i * kernel_size + j] * tile[(threadIdx.y + i) * BLOCK_SIZE + (threadIdx.x + j)];
            }
        }

        result[row * col_size + col] = temp;
    }
}

////////////////////////////////////////////////////////////////////////////////////
//Launch kernel on  multiple GPU
///////////////////////////////////////////////////////////////////////////////////
void solvePoissonEquation(cufftComplex* d_ft, cufftComplex* d_ft_k, float* k, int row_size, int col_size)
{
    const int BSZ_Y = 4;
    const int BSZ_X = 4;
    dim3 dimGrid (int(row_size/BSZ_X), int(col_size/BSZ_Y));
    dim3 dimBlock (BSZ_X, BSZ_Y);

    solvePoisson<<<dimGrid,dimBlock>>>(d_ft, d_ft_k, k, row_size, col_size, BSZ_Y);

    // Wait for device to finish all operation
    cudaDeviceSynchronize();

    // Check if kernel execution generated and error
    //getLastCudaError("Kernel execution failed [ solvePoisson ]");
    cudaError_t err = cudaGetLastError();   
    if ( err != cudaSuccess ){
            //fprintf(stderr, "Kernel execution failed [ solvePoisson ]\n");
            printf("CUDA Error: %s\n", cudaGetErrorString(err));   
            return;	
    }

}

////////////////////////////////////////////////////////////////////////////////
// Kernel for Solving Poisson equation on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void solvePoisson(cufftComplex* ft, cufftComplex* ft_k, float* k, int row_size, int col_size, int BSZ)
{
     int i = threadIdx.y + blockIdx.y * blockDim.y;
     int j = threadIdx.x + blockIdx.x * blockDim.x;
     int index = i*col_size+j;
     if (i<row_size && j<col_size)
     {
         float k2 = k[i]*k[i] + k[j]*k[j];
         if (i==0 && j==0)
         {
             k2 = 1.0f;
         }

         ft_k[index].x = -ft[index].x*1/k2;
         ft_k[index].y = -ft[index].y*1/k2;
    }
}


__global__ void multiply(cufftComplex* A, int row_size, int col_size, cufftComplex* B, cufftComplex* result, int batch_size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int index = i*col_size*batch_size+j;
    float a,b,c,d;
    if (i < row_size && j < batch_size*col_size) {
        a = A[index].x;
        b = A[index].y;
        c = B[index].x;
        d = B[index].y;

        result[index].x = a*c - b*d;
        result[index].y = a*d + b*c;
    }
}

void cmat_mult(cufftComplex* A, int row_size, int col_size, cufftComplex* B, cufftComplex* result, int batch_size){
    int block = 16;
    dim3 threadsPerBlock(block, block);
    dim3 blocksPerGrid(ceil(row_size/block)+1, ceil(col_size*batch_size/block)+1);

    multiply<<<blocksPerGrid, threadsPerBlock>>>(A, row_size, col_size, B, result, batch_size);

    // Wait for device to finish all operation
    cudaDeviceSynchronize();

    // Check if kernel execution generated and error
    //getLastCudaError("Kernel execution failed [ solvePoisson ]");
    cudaError_t err = cudaGetLastError();   
    if ( err != cudaSuccess ){
            //fprintf(stderr, "Kernel execution failed [ solvePoisson ]\n");
            printf("CUDA Error: %s\n", cudaGetErrorString(err));   
            return;	
    }
}


namespace convolution {


    vector<vector<vector<float>>> nvidia_convolve2d(vector<vector<vector<float>>>& images, vector<vector<float>>& kernel){
        
        int batch_size = images.size();

        int paddingSize = -1;
        int topPadding, bottomPadding;

        unsigned long long image_size [2] = {images[0].size(), images[0][0].size()};

        // error checking
        if(paddingSize < -1) throw invalid_argument("Padding size must be a non negative number.");

        // check if kernel is empty
        if(kernel.size() == 0 && kernel[0].size() == 0) throw invalid_argument("kernel must be non empty.");

        // check if image is empty
        if(images.size() == 0) throw invalid_argument("image must be non empty.");

        // ensure every row is the same length
        int kernel_size = kernel[0].size();
        for(int i = 1; i < kernel.size(); ++i){
            if(kernel[i].size() != kernel_size) throw invalid_argument("kernel must be a rectangular matrix.");
        }

        // end error checking

        // calculate padding size
        if(paddingSize == -1) {

            topPadding = ceil((kernel.size()-1) / 2.);
            bottomPadding = floor((kernel[0].size()-1) / 2.);

        } else {
            topPadding = paddingSize;
            bottomPadding = paddingSize;
        }

        int padding = topPadding + bottomPadding;

        // calculate new size of image based on padding size
        int row_size = (topPadding + bottomPadding + image_size[0]);
        int col_size = (topPadding + bottomPadding + image_size[1]);
        int size = row_size * col_size;

        // allocate space for linear indexed arrays
        Complex* linear_image = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* result = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* linear_kernel = (Complex*)malloc(size * batch_size * sizeof(Complex));

        auto linear_start = chrono::high_resolution_clock::now();

        int index, batch_idx;
        int batch = 0;
        // create linear indexed image with padding
        //for (int batch = 0; batch < batch_size; ++batch){
        for(auto& image: images){
            batch_idx = batch*size;
            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {
                    index = batch_idx + (i*col_size + j);
                    linear_image[index].y = 0.f;
                    if (i < image_size[0] && j < image_size[1]) { 
                        linear_image[index].x = image[i][j];
                    } else {
                        linear_image[index].x = 0.f; // add padding
                    }
                }
            }
            ++batch;
        }

        // create linear indexed filter
        int kernel_col_size = kernel[0].size();
        for (int batch = 0; batch < batch_size; ++batch){
            batch_idx = batch*size;
            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {
                    index = batch_idx + (i*col_size + j);
                    linear_kernel[index].y = 0;
                    if (i < kernel.size() && j < kernel[0].size()) {
                        linear_kernel[index].x = kernel[i][j];
                    } else {
                        linear_kernel[index].x = 0.f;
                    }
                }
            }
        }

        cufftComplex* d_image;
        cufftComplex* d_result;
        cufftComplex* d_kernel;

        int n[2] = {row_size, col_size};
    
        cudaMalloc((void**)&d_image, sizeof(cufftComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }

        cudaMalloc((void**)&d_result, sizeof(cufftComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        cudaMalloc((void**)&d_kernel, sizeof(cufftComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, batch*size*sizeof(cufftComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, linear_kernel, batch*size*sizeof(cufftComplex), cudaMemcpyHostToDevice);

        cufftHandle plan;
        cufftHandle plan_k;
        int idist = size;
        int odist = size;
        
        int inembed[] = {row_size, col_size};
        int onembed[] = {row_size, col_size};

        int istride = 1;
        int ostride = 1;

        if (cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }
        if (cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }

        if (cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cufftExecC2C(plan_k, d_kernel, d_kernel, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cudaDeviceSynchronize() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to synchronize\n");
            //return;
        }
        
        // element-wise multiplication of the image and kernel
        cmat_mult(d_image, row_size, col_size, d_kernel, d_result, batch_size);

        // transform out of fourier space
        if (cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        // copy results from device to host
        cudaMemcpy(result, d_result, batch*size*sizeof(Complex), cudaMemcpyDeviceToHost);

        // initialize return vector
        vector<vector<float>> out( row_size , vector<float> (col_size, 0));
        
        // add results to 2d vector to return
        vector<vector<vector<float>>> double_out = {};
        for(int batch = 0; batch < batch_size; ++batch) {
            batch_idx = size*batch;
            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {
                    //cout << result[batch_idx + i*col_size + j].x << " ";
                    out[i][j] = (result[batch_idx + i*col_size + j].x/(size));
                }
            }
            double_out.push_back(out);
        }
    
        // free host memory
        free(linear_image);
        free(result);
        free(linear_kernel);

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return double_out;

    }

    vector<vector<float>> tile_convolve2d(vector<vector<float>>& image, vector<vector<float>>& kernel){
        int paddingSize = -1;
        int topPadding, bottomPadding;
        unsigned long long image_size [2] = {image.size(), image[0].size()};

        // error checking
        if(paddingSize < -1) throw invalid_argument("Padding size must be a non negative number.");

        // check if kernel is empty
        if(kernel.size() == 0 && kernel[0].size() == 0) throw invalid_argument("kernel must be non empty.");

        // check if image is empty
        if(image.size() == 0 && image[0].size() == 0) throw invalid_argument("image must be non empty.");

        // ensure every row is the same length
        for(int i = 1; i < image.size(); ++i){
            if(image[i].size() != image_size[1]) throw invalid_argument("image must be a rectangular matrix.");
        }

        // ensure every row is the same length
        int kernel_size = kernel[0].size();
        for(int i = 1; i < kernel.size(); ++i){
            if(kernel[i].size() != kernel_size) throw invalid_argument("kernel must be a rectangular matrix.");
        }

        // end error checking

        // calculate padding size
        if(paddingSize == -1) {

            topPadding = ceil((kernel.size()-1) / 2.);
            bottomPadding = floor((kernel[0].size()-1) / 2.);

        } else {
            topPadding = paddingSize;
            bottomPadding = paddingSize;
        }

        // calculate new size of image based on padding size
        int row_size = (topPadding + bottomPadding + image_size[0]);
        int col_size = (topPadding + bottomPadding + image_size[1]);
        int size = row_size * col_size;

        // allocate space for linear indexed arrays
        float* linear_image = (float*)malloc(size * sizeof(float));
        float* result = (float*)malloc(size * sizeof(float));
        float* linear_kernel = (float*)malloc(kernel.size()*kernel[0].size() * sizeof(float));

        auto linear_start = chrono::high_resolution_clock::now();

        int index;
        // create linear indexed image with padding
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = i*col_size + j;
                if (j < bottomPadding || j >= bottomPadding + image_size[1]) {
                    linear_image[index] = 0;
                } else if (i < bottomPadding || i >= bottomPadding + image_size[0]) {
                    linear_image[index] = 0;
                } else {
                    linear_image[index] = image[i-bottomPadding][j-bottomPadding];
                }
            }
        }

        // create linear indexed filter
        for (int i = 0; i < kernel.size(); ++i) {
            for (int j = 0; j < kernel[0].size(); ++j) {
                linear_kernel[i*kernel.size() + j] = kernel[kernel.size() - 1 - i][kernel.size() - 1 - j];
            }
        }

        auto linear_end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> linear_ms = linear_end - linear_start;
        //cout << "time taken to flatten array: " << linear_ms.count() << endl;

        float* d_image;
        float* d_result;
        float* d_kernel;

        // allocate vectors for GPU
        cudaMalloc(&d_image, size*sizeof(float));
        
        cudaMalloc(&d_result, size*sizeof(float));
        cudaMalloc(&d_kernel, kernel.size()*kernel.size()*sizeof(float));
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, linear_kernel, kernel.size()*kernel.size()*sizeof(float), cudaMemcpyHostToDevice);


        int tile_width = BLOCK_SIZE - (kernel.size() -1);
        
        dim3 num_blocks(ceil(col_size / (float) tile_width), ceil(row_size / (float) tile_width), 1);
        dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE, 1);

        printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
        num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x, num_threads.y, num_threads.z);

        int offset = kernel.size() / 2.; // center of filter
        
        // call kernel
        auto start = chrono::high_resolution_clock::now();
        //convolve2d_helper_opt<<<num_blocks, num_threads>>>(d_image, row_size, col_size, d_kernel, kernel.size(), offset, d_result, tile_width);
        convolve2d_helper_opt<<<num_blocks, num_threads>>>(d_image, row_size, col_size, d_kernel, kernel.size(), offset, d_result, tile_width);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();

        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> time  = end - start;
        cout << "time in kernel: " << time.count() << endl;

        // copy results from device to host
        cudaMemcpy(result, d_result, size*sizeof(float), cudaMemcpyDeviceToHost);
        
        // initialize return vector
        vector<vector<float>> out( row_size , vector<float> (col_size, 0));
        
        // add results to 2d vector to return
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                out[i][j] = result[i*col_size + j];
            }
        }

        // free host memory
        free(linear_image);
        free(result);
        free(linear_kernel);

        // free device memory
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return out;
    }

    vector<vector<float>> convolve2d(vector<vector<float>>& image, vector<vector<float>>& kernel, int threads, int paddingSize){

        int topPadding, bottomPadding;
        unsigned long long image_size [2] = {image.size(), image[0].size()};

        // error checking
        if(paddingSize < -1) throw invalid_argument("Padding size must be a non negative number.");

        // check if kernel is empty
        if(kernel.size() == 0 && kernel[0].size() == 0) throw invalid_argument("kernel must be non empty.");

        // check if image is empty
        if(image.size() == 0 && image[0].size() == 0) throw invalid_argument("image must be non empty.");

        // ensure every row is the same length
        for(int i = 1; i < image.size(); ++i){
            if(image[i].size() != image_size[1]) throw invalid_argument("image must be a rectangular matrix.");
        }

        // ensure every row is the same length
        int kernel_size = kernel[0].size();
        for(int i = 1; i < kernel.size(); ++i){
            if(kernel[i].size() != kernel_size) throw invalid_argument("kernel must be a rectangular matrix.");
        }

        // end error checking

        // calculate padding size
        if(paddingSize == -1) {

            topPadding = ceil((kernel.size()-1) / 2.);
            bottomPadding = floor((kernel[0].size()-1) / 2.);

        } else {
            topPadding = paddingSize;
            bottomPadding = paddingSize;
        }

        // calculate new size of image based on padding size
        int row_size = (topPadding + bottomPadding + image_size[0]);
        int col_size = (topPadding + bottomPadding + image_size[1]);
        int size = row_size * col_size;

        // allocate space for linear indexed arrays
        float* linear_image = (float*)malloc(size * sizeof(float));
        float* result = (float*)malloc(size * sizeof(float));
        float* linear_kernel = (float*)malloc(kernel.size()*kernel[0].size() * sizeof(float));

        auto linear_start = chrono::high_resolution_clock::now();

        int index;
        // create linear indexed image with padding
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = i*col_size + j;
                if (j < topPadding || j >= topPadding + image_size[1]) {
                    linear_image[index] = 0;
                } else if (i < topPadding || i >= topPadding + image_size[0]) {
                    linear_image[index] = 0;
                } else {
                    linear_image[index] = image[i-topPadding][j-topPadding];
                }
            }
        }

        // create linear indexed filter
        for (int i = 0; i < kernel.size(); ++i) {
            for (int j = 0; j < kernel[0].size(); ++j) {
                linear_kernel[i*kernel.size() + j] = kernel[i][j];
            }
        }

        auto linear_end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> linear_ms = linear_end - linear_start;
        //cout << "time taken to flatten array: " << linear_ms.count() << endl;

        float* d_image;
        float* d_result;
        float* d_kernel;

        // allocate vectors for GPU
        cudaMalloc(&d_image, size*sizeof(float));
        cudaMalloc(&d_result, size*sizeof(float));
        cudaMalloc(&d_kernel, kernel.size()*kernel.size()*sizeof(float));
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, linear_kernel, kernel.size()*kernel.size()*sizeof(float), cudaMemcpyHostToDevice);

        //cout << "threads: " << threads << endl;
        int X_THREADS = threads;
        int Y_THREADS = X_THREADS;
        int X_BLOCKS = (col_size + X_THREADS - 1) / X_THREADS;
        int Y_BLOCKS = (row_size + Y_THREADS - 1) / Y_THREADS;

        dim3 block_dim(X_THREADS, Y_THREADS);
        dim3 grid_dim(X_BLOCKS, Y_BLOCKS);

        int offset = kernel.size() / 2.; // center of filter
        
        // call kernel
        auto start = chrono::high_resolution_clock::now();
        convolve2d_helper<<<grid_dim, block_dim>>>(d_image, row_size, col_size, d_kernel, kernel.size(), offset, d_result);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();

        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> time  = end - start;
        //cout << "time in kernel: " << time.count() << endl;

        // copy results from device to host
        cudaMemcpy(result, d_result, size*sizeof(float), cudaMemcpyDeviceToHost);
        
        // initialize return vector
        vector<vector<float>> out( row_size , vector<float> (col_size, 0));
        
        // add results to 2d vector to return
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                out[i][j] = result[i*col_size + j];
            }
        }

        // free host memory
        free(linear_image);
        free(result);
        free(linear_kernel);

        // free device memory
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return out;
    }
}
