
#include "cuda_defs.h"

struct saxpy_gpu 
{
    const float a;

    saxpy_gpu (float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

#if 0
template<class Op>
__global__ void saxpy_kernel(const float *x, float *y, const unsigned int N,
		const unsigned int n_block,
		Op op )
{
	int index= (blockDim.x * blockIdx.x + threadIdx.x)*n_block;
	int i;
	for( i= 0; i < n_block; i++ )
		if( (index+i) < N )
			y[index+i]= op( x[index+i], y[index+i] );
}
#endif
