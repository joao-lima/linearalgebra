
#include "cuda_defs.h"

template<class Op>
__global__ void saxpy_kernel(const float *x, float *y, const unsigned int N,
		Op op )
{
	int index= blockDim.x * blockIdx.x + threadIdx.x;
	if( index < N )
		y[index]= op( x[index], y[index] );
}

struct saxpy_gpu 
{
    const float a;

    saxpy_gpu (float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};
