#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

namespace convolution{
    std::vector<std::vector<std::vector<float>>> nvidia_convolve2d(std::vector<std::vector<std::vector<float>>>& image, std::vector<std::vector<float>>& kernel);
    std::vector<std::vector<float>> tile_convolve2d(std::vector<std::vector<float>>& image, std::vector<std::vector<float>>& kernel);
    std::vector<std::vector<float>> convolve2d(std::vector<std::vector<float>>& image, std::vector<std::vector<float>>& kernel, int threads=16, int paddingSize=-1);

}