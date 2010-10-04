
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include "cublas.h"

#include "cuda_safe.h"

/* Host implementation of a simple version of sgemm */
static void
simple_sgemm(int n, float alpha, const float *A, const float *B,
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

/* Main */
int main(int argc, char **argv)
{
	cublasStatus status;
	float *h_A;
	float *h_B;
	float *h_C;
	float *h_C_ref;
	float *d_A = 0;
	float *d_B = 0;
	float *d_C = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int N= 1024;
	int n2 = N * N;
	int i;
	int max_iter= 10;
	float error_norm;
	float ref_norm;
	float diff;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;

	cudaStream_t stream[3];
	for( i= 0; i < 3; i++ )
		cudaStreamCreate( &stream[i] );
	/* Initialize CUBLAS */
	printf("simpleCUBLAS test running..\n");

	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );

	if( argc > 1 ) {
		N= atoi( argv[1] );
		n2 = N * N;
	}

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	/* Allocate host memory for the matrices */
	//h_A = (float *)malloc(n2 * sizeof(h_A[0]));
	CUDA_SAFE_CALL( cudaHostAlloc((void**)&h_A, n2 * sizeof(h_A[0]),
			cudaHostAllocDefault) );
	if (h_A == 0) {
		fprintf(stderr, "!!!! host memory allocation error (A)\n");
		return EXIT_FAILURE;
	}
	CUDA_SAFE_CALL( cudaHostAlloc((void**)&h_B, n2 * sizeof(h_B[0]),
			cudaHostAllocDefault) );
	if (h_B == 0) {
		fprintf(stderr, "!!!! host memory allocation error (B)\n");
		return EXIT_FAILURE;
	}
	CUDA_SAFE_CALL( cudaHostAlloc((void**)&h_C, n2 * sizeof(h_C[0]),
			cudaHostAllocDefault) );
	if (h_C == 0) {
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	/* Fill the matrices with test data */
	for (i = 0; i < n2; i++) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
		h_C[i] = rand() / (float)RAND_MAX;
	}

	/* Allocate device memory for the matrices */
	status = cublasAlloc(n2, sizeof(d_A[0]), (void **)&d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error (A)\n");
		return EXIT_FAILURE;
	}
	status = cublasAlloc(n2, sizeof(d_B[0]), (void **)&d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error (B)\n");
		return EXIT_FAILURE;
	}
	status = cublasAlloc(n2, sizeof(d_C[0]), (void **)&d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	CUDA_SAFE_CALL(cudaEventRecord( e1, 0 ));
	for( i= 0; i < max_iter; i++ ){
		/* Initialize the device matrices with the host matrices */
		status = cublasSetVectorAsync(n2, sizeof(h_A[0]), h_A,
			       	1, d_A, 1, stream[0]);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! device access error (write A)\n");
			return EXIT_FAILURE;
		}
		status = cublasSetVectorAsync(n2, sizeof(h_B[0]), h_B, 1,
				d_B, 1, stream[1]);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! device access error (write B)\n");
			return EXIT_FAILURE;
		}
		status = cublasSetVectorAsync(n2, sizeof(h_C[0]), h_C, 1,
			       	d_C, 1, stream[2]);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! device access error (write C)\n");
			return EXIT_FAILURE;
		}

		cudaThreadSynchronize();
		/* Clear last error */
		cublasGetError();

		/* Performs operation using cublas */
		cublasSgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N, beta,
				d_C, N);
		status = cublasGetError();
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! kernel execution error.\n");
			return EXIT_FAILURE;
		}

		/* Read the result back */
		status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! device access error (read C)\n");
			return EXIT_FAILURE;
		}

	}
	CUDA_SAFE_CALL(cudaEventRecord( e2, 0 ));
	CUDA_SAFE_CALL(cudaEventSynchronize( e2 ));
	CUDA_SAFE_CALL(cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ));
	bandwidth_in_MBs= 1e3f * max_iter * (3.0f*N*N*sizeof(float)) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "size= %d time(s)= %.3f bandwidth(MB/s)= %.1f\n",
		N, elapsed_time_in_Ms/(1e3f*max_iter), bandwidth_in_MBs );

	if( argc > 2 ) {
		/* Allocate host memory for reading back the result from device memory */
		/* Performs operation using plain C code */
		h_C_ref = (float *)malloc(n2 * sizeof(h_C[0]));
		simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);
		if (h_C_ref == 0) {
			fprintf(stderr, "!!!! host memory allocation error (C)\n");
			return EXIT_FAILURE;
		}
		/* Check result against reference */
		error_norm = 0;
		ref_norm = 0;
		for (i = 0; i < n2; ++i) {
			diff = h_C_ref[i] - h_C[i];
			error_norm += diff * diff;
			ref_norm += h_C_ref[i] * h_C_ref[i];
		}
		error_norm = (float)sqrt((double)error_norm);
		ref_norm = (float)sqrt((double)ref_norm);
		if (fabs(ref_norm) < 1e-7) {
			fprintf(stderr, "!!!! reference norm is 0\n");
			return EXIT_FAILURE;
		}
		printf("Test %s\n",
		       (error_norm / ref_norm < 1e-6f) ? "PASSED" : "FAILED");

		free(h_C_ref);
	}

	/* Memory clean up */
	CUDA_SAFE_CALL( cudaFreeHost(h_A));
	CUDA_SAFE_CALL( cudaFreeHost(h_B));
	CUDA_SAFE_CALL( cudaFreeHost(h_C));
	status = cublasFree(d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error (A)\n");
		return EXIT_FAILURE;
	}
	status = cublasFree(d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error (B)\n");
		return EXIT_FAILURE;
	}
	status = cublasFree(d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error (C)\n");
		return EXIT_FAILURE;
	}

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
