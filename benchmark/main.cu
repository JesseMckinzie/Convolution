#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include "../src/CUDA/convolution.cuh"
#include "../src/Convolution.hpp"

using namespace std;
typedef vector<vector<float>> Matrix;

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

int main(){
    string path = "E:\\CUDA_benchmark_data\\data\\";

    //vector<int16 image_sizes {200, 400, 800, 1600, 3200, 6400, 12800, 25600};
    int size = 25600;

    vector<float> in;
    vector<string> vec;
    string str;
    string filePath;

    Matrix K;        
    filePath = path + "K10.txt";
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
    filePath = path + name;
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
    auto start = std::chrono::high_resolution_clock::now();
    vector<vector<float>> out = convolution::tile_convolve2d(img, K);
    //Matrix out = Convolution::convolution2d(img, K);
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> ms = end - start;
    cout << "Total time: " << ms.count() << endl;
    cout << endl;
    //img.clear();

    write("test_result.txt", out);
    
}
