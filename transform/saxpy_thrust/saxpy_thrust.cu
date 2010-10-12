
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "cuda_defs.h"

struct saxpy_gpu : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_gpu (float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_fast( float A, thrust::device_vector<float>& X,
		thrust::device_vector<float>& Y )
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(),
		    saxpy_gpu(A));
}

void randomInit( float* data, int size )
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main( int argc, char *argv[] )
{
	unsigned int mem_size= (1 << 25);
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	int i, max_iter= 10;
	float *x, *y;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));
	unsigned int nelem= mem_size/sizeof(float);

	x= (float*) malloc( mem_size );
	y= (float*) malloc( mem_size );
	randomInit( x, nelem );
	randomInit( y, nelem );

	cudaSetDevice( DEVICE );
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		// transfer to device
		thrust::device_vector<float> X(x, x + nelem);
		thrust::device_vector<float> Y(y, y + nelem);
		// fast method
		saxpy_fast( 2.0, X, Y );
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );

	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 3.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "saxpy n=%d size(MB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		nelem, mem_size/(1<<20), elapsed_time_in_Ms/max_iter,
	       	bandwidth_in_MBs );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	free( x );
	free( y );

	cudaThreadExit();
}

