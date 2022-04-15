#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <tuple>
#include "../src/CUDA/convolution.cuh"
#include "../src/Convolution.hpp"

using namespace std;
typedef vector<vector<float>> Matrix;

namespace fs = filesystem;

// start helper methods 

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

// end helper methods

<<<<<<< Updated upstream
tuple<double, double> run_benchmark(int size, int kernel_size, string& image_path, string& kernel_path){
=======
int main(){
    string path = "/home/ec2-user/Dev/Convolution/Convolution/data";

    //vector<int16 image_sizes {200, 400, 800, 1600, 3200, 6400, 12800, 25600};
    int size = 25600;
>>>>>>> Stashed changes

    vector<float> in;
    vector<string> vec;
    string str;
    string filePath;

    Matrix K;        
    filePath = kernel_path;
    //cout << "filePath: " << filePath << endl;
    ifstream infile1;
    infile1.open(filePath);
    //cout << "after file" << endl;
    if(!infile1){
        cout << "Invalid path " + filePath << endl; 
        exit(1);
    }
   // cout << "after" << endl;
    while(getline(infile1, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        K.push_back(in);
        in.clear();
    }

    Matrix img;
    ifstream infile2;
    
    string name = "img";
    name += std::to_string(size);
    name += ".txt";
    filePath = image_path + "img" + std::to_string(size) + ".txt";
    cout << "filepath: " << filePath << endl;
    infile2.open(filePath);
    if(!infile2){
        cout << "Invalid path " + filePath << endl; 
        exit(1);
    }
    while(getline(infile2, str)){
        //cout << str << endl;
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        img.push_back(in);
        in.clear();
    }
    cout << "Image is size: " << std::to_string(size) << " running." << endl;
    cout << "Kernel is size: " << std::to_string(kernel_size) << endl;

    auto start = std::chrono::high_resolution_clock::now();
    vector<vector<float>> out = convolution::convolve2d(img, K);
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> cuda_ms = end - start;
    cout << "cuda time: " << cuda_ms.count() << endl;

    start = std::chrono::high_resolution_clock::now();
    Matrix cpp_out = Convolution::convolution2d(img, K);
    end = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> cpp_ms = end - start;
    cout << "cpp time: " << cpp_ms.count() << endl;

    double cpp_time = cpp_ms.count();
    double cuda_time = cuda_ms.count();


    cout << endl;
    //img.clear();
    //write("test_result.txt", out);
    return make_tuple(cpp_time, cuda_time);
}

int get_kernel_size(string name){
    string num;
    for(int i = 1; i < name.length(); ++i){
        if(isdigit(name[i])) {
            num+=name[i];
        }
        else break;
    }
    return stoi(num);
}

int main(){
    vector<int> sizes {256, 1256, 2256, 3256, 4256, 5256};

    ofstream outfile("times_1.txt");
    tuple<double, double> times;
    for(const auto& size: sizes){
        outfile << std::to_string(size) << ";";

        string image_path = "E:\\CUDA_benchmark_data\\new\\" + std::to_string(size) + "\\";
        fs::directory_iterator directory_iterator = fs::directory_iterator(image_path + "kernel");
        vector<int> kernel_sizes;
        string kernel_basename;
        vector<string> kernel_paths;
        for(const auto& p: directory_iterator){
            kernel_paths.push_back(p.path().string());
            kernel_basename = p.path().filename().string();
            kernel_sizes.push_back(get_kernel_size(kernel_basename));
        }
        int kernel_size;
        string kernel_path;

        //sort(kernel_sizes.begin(), kernel_sizes.end());

        for(int i = 0; i < kernel_paths.size(); ++i){
            kernel_size = kernel_sizes[i];
            kernel_path = kernel_paths[i];
            outfile << "kernel " << std::to_string(kernel_size) << ";";
            times = run_benchmark(size, kernel_size, image_path, kernel_path);
            outfile << "cpp " + std::to_string(get<0>(times)) + ";";
            outfile << "cuda " + std::to_string(get<1>(times)) << ";";
            cout << endl;
        }
        outfile << "\n";

        //string kernel_path = image_path + "kernel\\K" + std::to_string(kernel_size) + "_" + std::to_string(size) + ".txt";

        //run_benchmark(size, 32, image_path, kernel_path);
    }
    outfile.close();
    
}
