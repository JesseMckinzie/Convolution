#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

namespace convolution{

    std::vector<std::vector<float>> convolve2d(std::vector<std::vector<float>>& image, std::vector<std::vector<float>>& kernel, int paddingSize=-1);

}