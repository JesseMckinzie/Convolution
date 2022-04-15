#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../src/CUDA/convolution.cuh"

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

    Matrix K;
    Matrix img;
    string str;
    vector<string> vec;
    vector<float> in;
    ifstream infile1("K.txt");
    while(getline(infile1, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        K.push_back(in);
        in.clear();
    }

    ifstream infile2("img.txt");
    while(getline(infile2, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        img.push_back(in);
        in.clear();
    }

    cout << "image" << endl;
    for(const auto vec:img){
        for(const auto v:vec){
            cout << v << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "kernel" << endl;
    for(const auto vec: K){
        for( const auto v:vec){
            cout << v << " ";
        }
        cout << endl;
    }

    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<float>> out1 = convolution::convolve2d(img, K);

    vector<vector<float>> out2 = convolution::tile_convolve2d(img, K);

    auto end = std::chrono::high_resolution_clock::now();

    chrono::duration<double, std::milli> ms = end - start;
    cout << "Total time: " << ms.count() << endl;

    write("result_act.txt", out1);
    write("result_opt.txt", out2);
}
