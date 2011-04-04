
#define CUDA_SAFE_CALL(call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }while(0)


#define	NTASKS	8
#define MAX_MEM		25	/* 32MB */
#define GRID_SIZE	64
#define	BLOCK_SIZE	512

__global__ void add1( float* array, unsigned int size )
{
  const unsigned int per_thread = size / (gridDim.x * blockDim.x);
  unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x) * per_thread;

  unsigned int j = size;
  if ( (blockIdx.x*blockDim.x + threadIdx.x) != (gridDim.x * blockDim.x-1) )
	  j = i + per_thread;

  for (; i < j; ++i)
	  ++array[i];
}

int check( const float *data, const unsigned int n, const float v )
{
	for( int i= 0; i < n; i++ )
		if( data[i] != v )
			return 1;

	return 0;
}

