#include "Convolution.hpp"
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

void write(const string& name, Matrix& vec){
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

    Matrix out = Convolution::convolution2d(img, K);

    write("result.txt", out);

}