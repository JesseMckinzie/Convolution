#include "Convolution.hpp"
#include <iostream>

using namespace std;

Matrix Convolution::convolution2d(Matrix& image, 
                                Matrix& kernel){
    Matrix out(image.size(), vector<float>(image[0].size(), 0));

    int xKSize = kernel.size();
    int yKSize = kernel[0].size();

    int kernelCenterX = xKSize / 2;
    int kernelCenterY = yKSize / 2;

    int ikFlip, jkFlip;
    int ii, jj;
    
    for(int i = 0; i < image.size(); ++i){
        for(int j = 0; j < image[0].size(); ++j){
            for(int ik = 0; ik < xKSize; ++ik){

                ikFlip = xKSize - 1 - ik;
                for(int jk = 0; jk < yKSize; ++jk){
                    jkFlip = yKSize - 1 - jk;

                    ii = i + (kernelCenterX - ikFlip);
                    jj = j + (kernelCenterY - jkFlip);

                    if(ii >= 0 && ii < image.size() && jj >= 0 && jj < image[0].size()){
                        out[i][j] += image[ii][jj] * kernel[ikFlip][jkFlip];
                    }
                }
            }
        }
    }
    
    return out;
}