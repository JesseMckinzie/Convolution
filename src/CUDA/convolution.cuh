#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

namespace convolution{

    std::vector<std::vector<float>> convolve2d(std::vector<std::vector<float>>& image, vector<vector<float>>& kernel, int paddingSize=-1);

}