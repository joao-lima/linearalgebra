#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

// Thread block size
#define BLOCK_SIZE 64
#ifndef DEVICE
#define DEVICE	0
#endif

__global__ void add_one( float *data, unsigned int nblock )
{
#if 1
	int index = (blockIdx.x*blockDim.x + threadIdx.x)*nblock;
	int i;
	float x;
	for( i= 0; i < nblock; i++ ){
		x= data[index+i];
		data[index+i] = x;
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
