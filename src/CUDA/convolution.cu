#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

typedef vector<vector<float>> Matrix;

__global__ void convolve2d_helper(float* image, int row_size, int col_size, float* kernel, int kernel_size, int kernel_offset, float* result){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row*col >= row_size*col_size) return;

    int start_r = row - kernel_offset;
    int start_c = col - kernel_offset;

    float temp = 0;

    int iFlip, jFlip;
    int ii, jj;
    for(int i = 0; i < kernel_size; ++i){
        iFlip = kernel_size - 1 - i;
        for(int j = 0; j < kernel_size; ++j){
            jFlip = kernel_size - 1 - j;

            ii = row + (kernel_offset - iFlip);
            jj = col + (kernel_offset - jFlip);

            //if((start_r + i) >= 0 && (start_r + i) < n && (start_c + j) >= 0 && (start_c + j) < n){
            if(ii >= 0 && ii < row_size && jj >= 0 && jj < col_size) {
                temp += image[ii * col_size + j] * kernel[i * kernel_size + j];
            }
        }
    }
    printf("temp: %d \n", row*col_size + col);
    result[row * col_size + col] = temp;
}


using namespace std;

vector<vector<float>> convolve2d(vector<vector<float>>& image, vector<vector<float>>& kernel, int paddingSize=-1){
    int topPadding, bottomPadding;
    unsigned long long image_size [2] = {image.size(), image[0].size()};

    if(paddingSize == -1) {
        topPadding = ceil((kernel.size()-1) / 2.);
        bottomPadding = floor((kernel.size()-1) / 2.);
    } else {
        topPadding = paddingSize;
        bottomPadding = paddingSize;
    }

    int row_size = (topPadding + bottomPadding + image_size[0]);
    int col_size = (topPadding + bottomPadding + image_size[1]);
    int size = row_size * col_size;

    float* linear_image = (float*)malloc(size * sizeof(float));
    float* result = (float*)malloc(size * sizeof(float));
    float* linear_kernel = (float*)malloc(kernel.size()*kernel.size() * sizeof(float));

    int index;
    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j){
            index = i*col_size + j;
            if(j < topPadding || j >= topPadding + image_size[1]){
                linear_image[index] = 0;
            } else if(i < bottomPadding || i >= bottomPadding + image_size[0]){
                linear_image[index] = 0;
            } else{
                linear_image[index] = image[i-bottomPadding][j-topPadding];
            }
        }
    }
   
    for(int i = 0; i < kernel.size(); ++i){
        for(int j = 0; j < kernel.size(); ++j){\
            linear_kernel[i*kernel.size() + j] = kernel[i][j];
        }
    }

    float* d_image;
    float* d_result;
    float* d_kernel;

    cudaMalloc(&d_image, size*sizeof(float));
    cudaMalloc(&d_result, size*sizeof(float));
    cudaMalloc(&d_kernel, kernel.size()*kernel.size()*sizeof(float));

    cudaMemcpy(d_image, linear_image, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, linear_kernel, kernel.size()*kernel.size()*sizeof(float), cudaMemcpyHostToDevice);

    int THREADS = 16;
    int BLOCKS = (size + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    int offset = kernel.size() / 2.;
    cout << "size: " << size << endl;
    cout << "kernel size: " << kernel.size() << endl;
    cout << "offset: " << offset << endl;
    cout << "blocks: " << BLOCKS << endl;
    cout << "threads: " << THREADS << endl;
      
    convolve2d_helper<<<grid_dim, block_dim>>>(d_image, row_size, col_size, d_kernel, kernel.size(), offset, d_result);

    cudaMemcpy(result, d_result, size*sizeof(float), cudaMemcpyDeviceToHost);

    vector<vector<float>> out( row_size , vector<float> (col_size, 0));

    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j){
            cout << result[i*col_size + j] << " ";
            out[i][j] = result[i*col_size + j];
        } cout << endl;
    }

    return out;

    /*
    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j){
            cout << linear_image[i*col_size + j] << " ";
        } cout << endl;
    }
    */

}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

vector<string> SplitString(string s){
	vector<string> v;
	string temp = "";
	for(int i=0;i<s.length();++i){
		
		if(s[i]==' '){
			v.push_back(temp);
			temp = "";
		}
		else{
			temp.push_back(s[i]);
		}
		
	}
	v.push_back(temp);
	
    return v;
}

void print(Matrix& vec){
    for(const auto& i: vec){
        for(const auto& j: i){
            cout << j << ' ';
        }
        cout << endl;
    }
}

void write(const string& name, vector<vector<float>>& vec){
    ofstream out(name);
    for(const auto& i: vec){
        for(const auto& j: i){
            out << j << ' ';
        }
        out << '\n';
    }
}

int main(){

    Matrix K;
    Matrix img;
    string str;
    vector<string> vec;
    vector<float> in;
    ifstream infile1("../K.txt");
    while(getline(infile1, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        K.push_back(in);
        in.clear();
    }

    ifstream infile2("../img.txt");
    while(getline(infile2, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        img.push_back(in);
        in.clear();
    }
    vector<vector<float>> out = convolve2d(img, K);

    write("../result.txt", out);

}



/*
int main(){
    //void convolve2d(float** image, int* image_size, float** kernel, int kernel_size, int paddingSize=-1)
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
