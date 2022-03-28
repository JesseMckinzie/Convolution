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

namespace convolution {
    vector<vector<float>> convolve2d(vector<vector<float>>& image, vector<vector<float>>& kernel, int paddingSize=-1){
        int topPadding, bottomPadding;
        unsigned long long image_size [2] = {image.size(), image[0].size()};

        // calculate padding size
        if(paddingSize == -1) {

            topPadding = ceil((kernel.size()-1) / 2.);
            bottomPadding = floor((kernel.size()-1) / 2.);

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
        float* linear_kernel = (float*)malloc(kernel.size()*kernel.size() * sizeof(float));

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
            for (int j = 0; j < kernel.size(); ++j) {
                linear_kernel[i*kernel.size() + j] = kernel[i][j];
            }
        }

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

        int X_THREADS = 16;
        int Y_THREADS = X_THREADS;
        int X_BLOCKS = (col_size + THREADS - 1) / THREADS;
        int Y_BLOCKS = (row_size + THREADS - 1) / THREADS;

        dim3 block_dim(X_THREADS, Y_THREADS);
        dim3 grid_dim(X_BLOCKS, Y_BLOCKS);

        int offset = kernel.size() / 2.; // center of filter
        
        // call kernel
        convolve2d_helper<<<grid_dim, block_dim>>>(d_image, row_size, col_size, d_kernel, kernel.size(), offset, d_result);

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

        return out;
    }
}
