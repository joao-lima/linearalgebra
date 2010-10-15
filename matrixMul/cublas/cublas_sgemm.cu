
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cublas.h"
#include "cuda_safe.h"

void randomInit(float*, int);

extern "C"
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	unsigned int N= 0;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	unsigned int i, max_iter= 10;
	float alpha = 1.0f;
	float beta = 0.0f;
	float error_norm;
	float ref_norm;
	float diff;

	if( argc > 1 )
		N = atoi( argv[1] );
	else
		N = 1024;

	cudaSetDevice( 0 );
    	CUBLAS_SAFE_CALL( cublasInit() );

	// allocate host memory
	unsigned int size_A = N * N;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*) malloc(mem_size_A);
	unsigned int size_B = N * N;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*) malloc(mem_size_B);
	unsigned int size_C = N * N;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*) malloc(mem_size_C);
	// initialize host memory
	srand(2006);
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);
	randomInit(h_C, size_C);

	// allocate CUBLAS device memory 
	float *d_A, *d_B, *d_C;
	CUBLAS_SAFE_CALL( cublasAlloc( size_A, sizeof(d_A[0]), (void**)&d_A) );
	CUBLAS_SAFE_CALL( cublasAlloc( size_B, sizeof(d_B[0]), (void**)&d_B) );
	CUBLAS_SAFE_CALL( cublasAlloc( size_C, sizeof(d_C[0]), (void**)&d_C) );
	CUBLAS_SAFE_CALL( cublasSetVector( size_A, sizeof(h_A[0]), h_A, 1, d_A, 1 ) );
	CUBLAS_SAFE_CALL( cublasSetVector( size_B, sizeof(h_B[0]), h_B, 1, d_B, 1 ) );
	CUBLAS_SAFE_CALL( cublasSetVector( size_C, sizeof(h_C[0]), h_C, 1, d_C, 1 ) );

	// events
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );

	CUDA_SAFE_CALL(cudaEventRecord( e1, 0 ));
	for( i= 0; i < max_iter; i++ ){
		/* Performs operation using cublas */
		cublasSgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N,
				beta, d_C, N);
		CUBLAS_SAFE_THREAD_SYNC();
	}
	CUDA_SAFE_CALL(cudaEventRecord( e2, 0 ));
	CUDA_SAFE_CALL(cudaEventSynchronize( e2 ));
	CUDA_SAFE_CALL(cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ));
	bandwidth_in_MBs= 1e3f * max_iter * (3.0f*N*N*sizeof(float)) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "size= %d time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		N, elapsed_time_in_Ms/(max_iter), bandwidth_in_MBs );


	if( argc > 2 ){
		// compute reference solution
		float* h_C_ref= (float*) malloc(mem_size_C);
		simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);

		/* Check result against reference */
		error_norm = 0;
		ref_norm = 0;
		for (i = 0; i < N; ++i) {
			diff = h_C_ref[i] - h_C[i];
			error_norm += diff * diff;
			ref_norm += h_C_ref[i] * h_C_ref[i];
		}
		error_norm = (float)sqrt((double)error_norm);
		ref_norm = (float)sqrt((double)ref_norm);
		if (fabs(ref_norm) < 1e-7) {
			fprintf (stderr, "!!!! reference norm is 0\n");
			return EXIT_FAILURE;
		}
		printf( "Test %s\n",
			(error_norm / ref_norm < 1e-6f) ? "PASSED" : "FAILED");

		free(h_C_ref);
	}

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	CUBLAS_SAFE_CALL( cublasFree(d_A) );
	CUBLAS_SAFE_CALL( cublasFree(d_B) );
	CUBLAS_SAFE_CALL( cublasFree(d_C) );
	CUDA_SAFE_CALL( cudaEventDestroy(e1) );
	CUDA_SAFE_CALL( cudaEventDestroy(e2) );

	CUBLAS_SAFE_CALL( cublasShutdown() );
	cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (data1[k] != data2[k]) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf(" nTotal Errors = %d n", error_count);
}

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;
            for (k = 0; k < n; ++k) {
                prod += A[k * n + i] * B[j * n + k];
            }
            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

