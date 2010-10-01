#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

// Thread block size
#define BLOCK_SIZE 256

__global__ void add_one( float *data )
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  float res;
  float x= data[index];
  res = x + 1;
        __syncthreads();
  data[index] = res;
}


__host__ int check( const float *data, const float v, const unsigned int n )
{
	int i;

	for( i= 0; i < n; i++ )
		if( data[i] != v )
			return 0;
	return 1;
}

#endif // #ifndef _ADD_KERNEL_H_
