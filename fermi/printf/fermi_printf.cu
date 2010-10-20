
#include <stdio.h>

__global__ void foo_kernel( unsigned int out )
{
	printf( "out kernel value=%d idx=%d\n", out, threadIdx.x );
}

int main( int argc, char **argv )
{
	cudaSetDevice(0);
	foo_kernel<<<1, 5>>>( 10 );
	cudaThreadExit();
	return 0;
}
