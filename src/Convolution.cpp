#include "Convolution.hpp"
#include <iostream>
#include <cmath>

using namespace std;

Matrix Convolution::convolution2d(Matrix& image, 
                                Matrix& kernel,
                                int paddingSize){

    int topPadding, bottomPadding;

    if(paddingSize == -1) {
        topPadding = ceil((kernel.size()-1) / 2.);
        bottomPadding = floor((kernel[0].size()-1) / 2.);
    } else {
        topPadding = paddingSize;
        bottomPadding = paddingSize;
    }

    vector<float> rowPadding(topPadding + bottomPadding + image[0].size());

    
    for(int i = 0; i < image.size(); ++i){
        for(int j = 0; j < topPadding; ++j) {
            image[i].insert(image[i].begin(), 0);
        }
        for(int j = 0; j < bottomPadding; ++j){
            image[i].push_back(0);
        }
    }

    for(int j = 0; j < topPadding; ++j) {
        image.insert(image.begin(), rowPadding);
    }
    for(int j = 0; j < bottomPadding; ++j){
        image.push_back(rowPadding);
    }
    
    Matrix out(image.size(), vector<float>(image[0].size(), 0));

    int xKSize = kernel[0].size(); // number of columns
    int yKSize = kernel.size(); // number of rows

    int kernelCenterX = xKSize / 2.;
    int kernelCenterY = yKSize / 2.;

    int ikFlip, jkFlip;
    int ii, jj;

    for(int i = 0; i < image.size(); ++i){
        for(int j = 0; j < image[0].size(); ++j){
            for(int ik = 0; ik < yKSize; ++ik){
                ikFlip = yKSize - 1 - ik;
                for(int jk = 0; jk < xKSize; ++jk){
                    jkFlip = xKSize - 1 - jk;

                    ii = i + (kernelCenterY - ikFlip);
                    jj = j + (kernelCenterX - jkFlip);

                    if(ii >= 0 && ii < image.size() && jj >= 0 && jj < image[0].size()){
                        out[i][j] += image[ii][jj] * kernel[ikFlip][jkFlip];
                    }
                }
            }
        }
    }
    
    return out;
}
