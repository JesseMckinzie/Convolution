#include <cmath>
#include <iostream>
#include <vector>


__global__ void convolve2d_helper(float* image, float* kernel, int n, int kernel_size, int kernel_offset){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockidx.x * blockDim.x + threadIdx.x;

    int start_r = row - kernel_offset;
    int start_c = col - kernel_offset;

    int temp = 0;

    for(int i = 0; i < kernel_size; ++i){
        for(int j = 0; j < kernel_offset; ++j){
            if((start_r + i) >= 0 && (start_r + i) < n && (start_c + j) >= 0 && (start_c + j) < n){
                temp += image[(start_r + i) * n + (start_c + j)] * kernel[i * kernel_size + j];
            }
        }
    }

    result[row * n + col] = temp;
}


using namespace std;

void convolve2d(vector<float> image, vector<float> kernel, int paddingSize=-1){
    int topPadding, bottomPadding;
    int image_size[2] = {image.size(), image[0].size()};

    if(paddingSize == -1) {
        topPadding = ceil((kernel_size-1) / 2.);
        bottomPadding = floor((kernel_size-1) / 2.);
    } else {
        topPadding = paddingSize;
        bottomPadding = paddingSize;
    }

    int row_size = (topPadding + bottomPadding + image_size[0]);
    int col_size = (topPadding + bottomPadding + image_size[1]);
    int size = row_size * col_size;

    float* linear_image = (float*)malloc(size * sizeof(float));

    int index;
    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j){
            index = i*col_size + j;
            cout << index << endl;
            if(j < topPadding || j >= topPadding + image_size[1]){
                linear_image[index] = 0;
            } else if(i < bottomPadding || i >= bottomPadding + image_size[0]){
                linear_image[index] = 0;
            } else{
                linear_image[index] = image[i-bottomPadding][j-topPadding];
            }
        }
    }

    float* linear_kernel = (float*)malloc(kernel_size*kernel_size * sizeof(float));
    for(int i = 0; i < kernel_size; ++i){
        for(int j = 0; j < kernel_size; ++j){
            linear_kernel[i*kernel_size + j] = kernel[i][j];
        }
    }

    /*
    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j){
            cout << linear_image[i*col_size + j] << " ";
        } cout << endl;
    }
    */

}
/*
int main(){
    void convolve2d(float** image, int* image_size, float** kernel, int kernel_size, int paddingSize=-1)
    int SIZE = 2;
    float** image;
    
    image = new float* [SIZE];
    for(int i = 0; i < SIZE; ++i){
        image[i] = new float[SIZE];
    }

    image[0][0] = 1.;
    image[0][1] = 2.;
    image[1][0] = 3.;
    image[1][1] = 4.;

    float** kernel;
    
    kernel = new float* [SIZE];
    for(int i = 0; i < SIZE; ++i){
        kernel[i] = new float[SIZE];
    }
    
    kernel[0][0] = 5;
    kernel[0][1] = 5;
    kernel[1][0] = 5;
    kernel[1][1] = 5;

    int image_size[2] = {2, 2};
    
    convolve2d(image, image_size, kernel, 2, 2);
}
*/