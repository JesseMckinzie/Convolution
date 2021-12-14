#include <vector>

typedef std::vector<std::vector<float> > Matrix;

class Convolution{
    public:
    static Matrix convolution2d(Matrix&, Matrix&);
};