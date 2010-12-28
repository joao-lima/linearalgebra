
#define NUM_SMS (24)
#define NUM_THREADS_PER_SM (384)
#define NUM_THREADS_PER_BLOCK (192)
#define NUM_BLOCKS ((NUM_THREADS_PER_SM / NUM_THREADS_PER_BLOCK) * NUM_SMS)
#define NUM_ITERATIONS 99999
 
// 128 MAD instructions
#define FMAD128(a, b) \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
 
__global__ void gflops_heavy( unsigned int work, float *data, unsigned int N,
		unsigned int offset, clock_t *timer )
{
	int idx = (blockIdx.x*blockDim.x + threadIdx.x +
		blockIdx.y*blockDim.y + threadIdx.y + offset)%N;
	__threadfence();
	if( idx == 0 )
		timer[0]= clock();

	float b = 1.01f;

	for (int i = 0; i < work; i++)
	{
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
		FMAD128(data[idx], b);
	}
	data[idx] = data[idx] + b;

	__threadfence();
	if( idx == 0 )
		timer[1]= clock();
}

__global__ void gflops_light( unsigned int work, float *data, unsigned int N,
		unsigned int offset, clock_t *timer )
{
	int idx = (blockIdx.x*blockDim.x + threadIdx.x +
		blockIdx.y*blockDim.y + threadIdx.y + offset)%N;
	__threadfence();
	if( idx == 0 )
		timer[0]= clock();

	float a = data[idx];  // this ensures the mads don't get compiled out
	float b = 1.01f;

	for (int i = 0; i < work; i++)
	{
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
		FMAD128(a, b);
	}
	data[idx] = a + b;

	__threadfence();
	if( idx == 0 )
		timer[1]= clock();
}
