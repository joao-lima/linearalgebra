
#include <stdio.h>

__device__ unsigned int fibo( unsigned int n )
{
	if( n > 2 ) {
		return fibo(n-1)+fibo(n-2);
	} else
		return 1;
}

__global__ void fibo_kernel( unsigned int n )
{
	unsigned int res;
	res= fibo( n );
	printf( "fibo idx=%d n=%d - %d\n", threadIdx.x, n, res );
}

int main( int argc, char **argv )
{
	cudaSetDevice(0);
	fibo_kernel<<<1, 5>>>( 10 );
	cudaThreadExit();
	return 0;
}
