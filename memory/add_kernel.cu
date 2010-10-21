#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

// Thread block size
#define BLOCK_SIZE 16

#ifndef DEVICE
#define DEVICE	0
#endif

__global__ void kernel_offset( float *data, const unsigned int N )
{
#if 1
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float a= 0e0f;
	if( (x < N) && (y < N) ) {
		a= data[y*N+x];
		data[y*N+x]= a;
	}
#endif
}


__host__ int check( const float *data, const float v, const unsigned int n )
{
	unsigned int i;

	for( i= 0; i < n; i++ )
		if( data[i] != v ){
			fprintf(stdout,"%d %f\n",i,data[i]);
			return 0;
		}
	return 1;
}

#endif // #ifndef _ADD_KERNEL_H_
