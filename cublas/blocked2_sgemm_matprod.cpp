
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

extern "C" {
#include "cblas.h"
}
#include "lapacke.h"

#define CONFIG_USE_FLOAT 1
#include "test_types.h"

#define CUDA_REGISTER	1

static int do_check
	(const double_type* a, const double_type* b, double_type* c,
 	double_type* cfinal, unsigned int n)
{
  double_type* const tmp = (double_type*)calloc(n * n, sizeof(double_type));
  if (tmp == NULL) return -1;
  double_type alpha = (double_type) 1.0;
  double_type beta = (double_type) 1.0;

  lacpy( LAPACK_COL_MAJOR, ' ', n, n, c, n, tmp, n );

  cblas_gemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
	a, n, b, n, beta, tmp, n );

  unsigned int i, j, k;
#if 0

//  for (i = 0; i < n * n; ++i) tmp[i] = 0.;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      for (k = 0; k < n; ++k)
	tmp[i * n +  j] += a[i * n + k] * b[k * n + j];
    }
  }
#endif
  int res = -1;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      k = i * n + j;
      if (fabsf(cfinal[k] - tmp[k]) >= 0.01)
      {
	printf("ERROR invalid @%p,%u,%u %f != %f\n", c, i, j, cfinal[k], tmp[k]);
	goto on_error;
      }
    }
  }

  res = 0;

 on_error:
  free(tmp);

  return res;
}

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for slarnv() */

double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

int
main( int argc, char **argv )
{
    double_type alpha = (double_type) 1.0;
    double_type beta = (double_type) 1.0;
    int N     = 512;
    int verif= 0;
    double t0, t1;
    cublasStatus_t res;
    cudaError_t err;
    cublasOperation_t transa, transb;
    cublasHandle_t handle;
    int cuda_max_streams =  1;
    int idx_stream= 0;
    cudaStream_t stream[40];
    int i, j, k, nb= 64;

    if( argc > 1 )
	N = atoi(argv[1]);

    if( argc > 2 )
	nb = atoi( argv[2] );

    if( argc > 3 )
	cuda_max_streams = atoi( argv[3] );

    if( argc > 4 )
	verif = atoi( argv[4] );

    cudaSetDevice(0);
    res= cublasCreate( &handle );
    if( res != CUBLAS_STATUS_SUCCESS ) {
	    fprintf(stdout, "CUBLAS create error: %d\n", res );
	    fflush(stdout);
	    return -1;
    }
    int info_solution;
    size_t msize = N*N;
#ifdef CUDA_REGISTER
    double_type *A      = (double_type *)malloc(msize*sizeof(double_type));
    double_type *B      = (double_type *)malloc(msize*sizeof(double_type));
    double_type *C      = (double_type *)malloc(msize*sizeof(double_type));
#else
    double_type *A, *B, *C;
	cudaMallocHost( (void**)&A, msize*sizeof(double_type) );
	cudaMallocHost( (void**)&B, msize*sizeof(double_type) );
	cudaMallocHost( (void**)&C, msize*sizeof(double_type) );
#endif
    double_type *Cinit;
    if( verif )
	    Cinit  = (double_type *)malloc(msize*sizeof(double_type));
    transa = transb = CUBLAS_OP_N;

    /* Check if unable to allocate memory */
    if ( (!A) || (!B) || (!C) ){
        printf("Out of Memory \n ");
        return -2;
    }

#ifdef CUDA_REGISTER
    cudaHostRegister( A, msize*sizeof(double_type), cudaHostRegisterPortable );
    cudaHostRegister( B, msize*sizeof(double_type), cudaHostRegisterPortable );
    cudaHostRegister( C, msize*sizeof(double_type), cudaHostRegisterPortable );
#endif


    larnv( IONE, ISEED, msize, A );
    larnv( IONE, ISEED, msize, B );
    larnv( IONE, ISEED, msize, C );

    if( verif )
	    lacpy( LAPACK_COL_MAJOR, ' ', N, N, C, N, Cinit, N );

    double_type *d_A, *d_B, *d_C;
    cudaMalloc( (void**)&d_A, msize*sizeof(double_type) );
    cudaMalloc( (void**)&d_B, msize*sizeof(double_type) );
    cudaMalloc( (void**)&d_C, msize*sizeof(double_type) );
    fprintf( stdout, "main A=%p B=%p C=%p\n", A, B, C );
    fprintf( stdout, "main d_A=%p d_B=%p d_C=%p\n", d_A, d_B, d_C );
    fflush(stdout);

    for( i= 0; i < cuda_max_streams; i++ ) {
	err= cudaStreamCreate( &stream[i] );
		if (err != CUDA_SUCCESS) {
			fprintf(stdout, "CUDA stream error: %d\n", err );
			fflush(stdout);
		}
    }
//	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST);
//	cublasSetStream( handle, 0 );
//    nb = (N / nb) * nb;
//	cublasSetMatrix( N, N, sizeof(double_type), A, N, d_A, N );
//	cublasSetMatrix( N, N, sizeof(double_type), B, N, d_B, N );
//	cublasSetMatrix( N, N, sizeof(double_type), C, N, d_C, N );
     cudaDeviceSynchronize();
    t0 = get_elapsedtime();
    for( i= 0; i < N; i+= nb ) {
	for( j= 0; j < N; j+= nb ) {
	    for( k= 0; k < N; k+= nb ) {
		err= cudaMemcpy2DAsync(
			d_A+k*N+i,
			N * sizeof(double_type),
			A+k*N+i,
			N * sizeof(double_type),
			nb * sizeof(double_type),
			nb,
			cudaMemcpyHostToDevice,
			stream[ idx_stream ] );
		if (err != CUDA_SUCCESS) {
			fprintf(stdout, "CUDA stream error: %d\n", err );
			fflush(stdout);
		}
		err= cudaMemcpy2DAsync(
			d_B+j*N+k, 
			N * sizeof(double_type),
			B+j*N+k,
			N * sizeof(double_type),
			nb * sizeof(double_type),
			nb,
			cudaMemcpyHostToDevice,
			stream[ idx_stream ] );
		if (err != CUDA_SUCCESS) {
			fprintf(stdout, "CUDA stream error: %d\n", err );
			fflush(stdout);
		}
		err= cudaMemcpy2DAsync(
			d_C+j*N+i,
			N * sizeof(double_type),
			C+j*N+i, 
			N * sizeof(double_type),
			nb * sizeof(double_type),
			nb,
			cudaMemcpyHostToDevice,
			stream[ idx_stream ] );
		if (err != CUDA_SUCCESS) {
			fprintf(stdout, "CUDA stream error: %d\n", err );
			fflush(stdout);
		}

#if 0
		cublasSetMatrixAsync( nb, nb, sizeof(double_type), 
			A+k*N+i, N,
			d_A+k*N+i, N,
		       kaapi_cuda_HtoD_stream() );
		cublasSetMatrixAsync( nb, nb, sizeof(double_type),
			B+j*N+k, N,
			d_B+j*N+k, N, 
		       kaapi_cuda_HtoD_stream() );
		cublasSetMatrixAsync( nb, nb, sizeof(double_type),
			C+j*N+i, N,
			d_C+j*N+i, N,
		       kaapi_cuda_HtoD_stream());

#endif
		cublasSetStream( handle, stream[ idx_stream ] );
		res= cublasGemm( handle,
		    transa, transb , nb, nb, nb, &alpha,
		    d_A+k*N+i, N,
		    d_B+j*N+k, N,
		    &beta,
		    d_C+j*N+i, N);
//		fprintf(stdout,"i=%d j=%d k=%d idx=%d\n", i, j, k, idx_stream);
//		fflush(stdout);
#if 0
		if( res != CUBLAS_STATUS_SUCCESS ) {
		    fprintf(stdout, "CUBLAS error: %d\n", res );
		    fflush(stdout);
		 //   return -1;
		}
#endif
#if 0
		cublasGetMatrixAsync( nb, nb, sizeof(double_type),
		       	d_C+j*N+i, N, C+j*N+i, N, kaapi_cuda_DtoH_stream() );
#endif
		err= cudaMemcpy2DAsync(
			C+j*N+i,
			N * sizeof(double_type),
			d_C+j*N+i, 
			N * sizeof(double_type),
			nb * sizeof(double_type),
			nb,
			cudaMemcpyDeviceToHost,
			stream[ idx_stream ] );
		if (err != CUDA_SUCCESS) {
			fprintf(stdout, "CUDA stream error: %d\n", err );
			fflush(stdout);
		}

		idx_stream = ( idx_stream + 1 ) % cuda_max_streams;
	    }
	}
    }
     err= cudaDeviceSynchronize();
    if (err != CUDA_SUCCESS) {
	    fprintf(stdout, "CUDA stream error: %d\n", err );
	    fflush(stdout);
    }
      t1 = get_elapsedtime();
//	cublasGetMatrix( N, N, sizeof(double_type), d_C, N, C, N );

    double tdelta = t1 - t0;
    double gflops = 1.0e-9 * ((2.0 * N * N * N)/(t1-t0));
    printf("# method  nblocks   size   time      GFlop/s\n");
    printf("SGEMM %d %6d %10.10f %9.6f\n", (int)N, nb, tdelta, gflops);
    fflush(stdout);

    if( verif ) {
	    /* Check the solution */
	    info_solution = do_check( A, B, Cinit, C, N );

	    if (info_solution == 0) {
		printf("# TESTING SGEMM .............. PASSED !\n");
	    }
	    else {
		printf("# TESTING SGEMM ... FAILED !\n");
	    }
    	free(Cinit); 
    }
    for( i= 0; i < cuda_max_streams; i++ ) {
	cudaStreamDestroy( stream[i] );
    }

#ifdef CUDA_REGISTER
    cudaHostUnregister( A );
    cudaHostUnregister( B );
    cudaHostUnregister( C );
    free( A );
    free( B );
    free( C );
#else
    cudaFreeHost( A );
    cudaFreeHost( B );
    cudaFreeHost( C );
#endif
    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_C );
    cublasDestroy( handle );

    return 0;
}
