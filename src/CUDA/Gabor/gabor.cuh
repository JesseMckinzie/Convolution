#include <cufftXt.h>
#include <vector>
#include <stdexcept>

#define CUFFT_MAX_SIZE 1 << 27

typedef float2 Complex;
typedef cufftComplex CuComplex;

void conv_dud_gpu(
    double* C,  // must be zeroed before call
    const unsigned int* A,
    double* B,
    int na, int ma, int nb, int mb);

__global__ void multiply(
    CuComplex* A, 
    int row_size, 
    int col_size, 
    CuComplex* B, 
    CuComplex* result, 
    int batch_size);

void cmat_mult(
    CuComplex* A, 
    int row_size, 
    int col_size, 
    CuComplex* B, 
    CuComplex* result, 
    int batch_size);

