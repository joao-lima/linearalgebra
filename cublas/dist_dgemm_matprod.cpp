
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" {
#include "cblas.h"
}
#include "lapacke.h"
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#define	    MAX_GPU	8

#define CONFIG_USE_DOUBLE 1
#include "test_types.h"

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

static void transpose_inplace(double_type* a, int n)
{
  // a the matrix
  // n the dimension

  for (int i = 0; i < n; ++i)
  {
    for (int j = i + 1; j < n; ++j)
    {
      const int ij = i * n + j;
      const int ji = j * n + i;
      const double_type tmp = a[ij];
      a[ij] = a[ji];
      a[ji] = tmp;
    }
  }
}

int
main( int argc, char **argv )
{
    double_type alpha = (double_type) 1.0;
    double_type beta = (double_type) 1.0;
    int N     = 1024;
    int verif= 0;
    double t0, t1;
    cublasHandle_t handle[MAX_GPU];
    cublasStatus_t res;
    cudaError_t errno;
    cublasOperation_t transa, transb;
    cudaStream_t stream[MAX_GPU];
    int info_solution;
    //int i, j;
    int k;
    int nb, NB, ngpu;
    nb = 512;
    NB = 512;
    ngpu = 1;

    if( argc > 1 )
	N = atoi(argv[1]);
    if( argc > 2 )
	NB = atoi(argv[2]);
    if( argc > 3 )
	ngpu = atoi(argv[3]);
    if( argc > 4 )
      verif = atoi( argv[4] );

    size_t msize = N*N;
    double_type *A, *B, *C;
    cudaMallocHost( (void**)&A, msize*sizeof(double_type) );
    cudaMallocHost( (void**)&B, msize*sizeof(double_type) );
    cudaMallocHost( (void**)&C, msize*sizeof(double_type) );
//    double_type *A      = (double_type *)malloc(msize*sizeof(double_type));
//    double_type *B      = (double_type *)malloc(msize*sizeof(double_type));
//    double_type *C      = (double_type *)malloc(msize*sizeof(double_type));
    double_type *Cinit;
    if( verif )
	    Cinit  = (double_type *)malloc(msize*sizeof(double_type));
    transa= transb = CUBLAS_OP_N;

    /* Check if unable to allocate memory */
    if ( (!A) || (!B) || (!C) ){
        printf("Out of Memory \n ");
        return -2;
    }

    larnv(IONE, ISEED, msize, A);
    larnv(IONE, ISEED, msize, B);
    larnv(IONE, ISEED, msize, C);
//    transpose_inplace( A, N );
    fprintf(stdout, "A[0]=%f B[0]=%f C[0]=%f\n",
	    A[0], B[0], C[0]);
    fflush(stdout);

    if( verif )
	    lacpy( LAPACK_COL_MAJOR, ' ', N, N, C, N, Cinit, N );

    cudaFree(0);
    cudaThreadSynchronize();

    double_type *d_A[MAX_GPU], *d_B[MAX_GPU], *d_C[MAX_GPU];

      t0 = get_elapsedtime();

    int iNB, jNB, gpu;
    for( gpu= 0; gpu < ngpu; gpu++ ) {
	fprintf(stdout, "setting gpu %d\n", gpu );
	fflush(stdout);
	cudaSetDevice(gpu);
	size_t NBsize = N * NB;
	errno= cudaMalloc( (void**)&d_A[gpu], NB*NB*sizeof(double_type) );
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMalloc error: %d\n", gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	errno= cudaMalloc( (void**)&d_B[gpu], NBsize*sizeof(double_type) );
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMalloc error: %d\n", gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	errno= cudaMalloc( (void**)&d_C[gpu], NB*NB*sizeof(double_type) );
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMalloc error: %d\n", gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	errno= cudaStreamCreate( &stream[gpu] );
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaStreamCreate error: %d\n", gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	res= cublasCreate( &handle[gpu] );
	if( res != CUBLAS_STATUS_SUCCESS ) {
		fprintf(stdout, "CUBLAS error: %d\n", res );
		fflush(stdout);
		return -1;
	}
	res= cublasSetStream( handle[gpu], stream[gpu] );
	if( res != CUBLAS_STATUS_SUCCESS ) {
		fprintf(stdout, "cublasSetStream error: %d\n", res );
		fflush(stdout);
		return -1;
	}
//	cublasSetMatrixAsync( N, N, sizeof(double_type), A, N, d_A[gpu], N,
//	       stream[gpu]	);
//	cublasSetMatrixAsync( N, N, sizeof(double_type), B, N, d_B[gpu], N,
//	       stream[gpu]	);
//	cublasSetMatrixAsync( N, N, sizeof(double_type), C, N, d_C[gpu], N,
//	       stream[gpu]	);
    }

    for( iNB= 0; iNB < N; iNB += NB ) {
    for( jNB= 0; jNB < N; jNB += NB ) {
	gpu = (iNB/NB)%ngpu;
	fprintf(stdout, "%d: memcpy gpu lines A(%d) B(%d)\n", gpu,
	      iNB, jNB );
	fflush(stdout);
	cudaSetDevice(gpu);
	cublasSetMatrixAsync( N, NB, sizeof(double_type),
		B+jNB*N, N,
		d_B[gpu], N,
	       stream[gpu]	);
	cublasSetMatrixAsync( NB, NB, sizeof(double_type),
		C+jNB*N+iNB, N,
		d_B[gpu], NB,
	       stream[gpu]	);
#if 0
	errno = cudaMemcpy2DAsync(
		d_A[gpu],
		N*sizeof(double_type),
		A+iNB*N,
		N*sizeof(double_type),
		N,
		NB,
		cudaMemcpyHostToDevice,
		stream[gpu]
		);
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMemcpy2DAsync error: %d\n",
		    gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	errno = cudaMemcpy2DAsync(
		d_B[gpu],
		N*sizeof(double_type),
		B+jNB*N,
		N*sizeof(double_type),
		N,
		NB,
		cudaMemcpyHostToDevice,
		stream[gpu]
		);
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMemcpy2DAsync error: %d\n",
		    gpu, errno );
	    fflush(stdout);
	    return -1;
	}
	errno = cudaMemcpy2DAsync(
		d_C[gpu],
		NB*sizeof(double_type),
		C+iNB*N+jNB,
		N*sizeof(double_type),
		NB,
		NB,
		cudaMemcpyHostToDevice,
		stream[gpu]
		);
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaMemcpy2DAsync error: %d\n",
		    gpu, errno );
	    fflush(stdout);
	    return -1;
	}

#endif
	for( k=0; k < N; k+=NB ) {
	    fprintf(stdout, "%d: compute gpu lines A(%d) B(%d) k %d\n", gpu,
		  iNB, jNB, k, N );
	    fflush(stdout);
	    cublasSetMatrixAsync( NB, NB, sizeof(double_type),
		    A+k*N+iNB, N,
		    d_A[gpu], NB,
		   stream[gpu]	);
	    res= cublasGemm( handle[gpu],
		    transa, transb , NB, NB, NB, &alpha,
		    d_A[gpu], NB,
		    d_B[gpu]+k, N,
		    &beta,
		    d_C[gpu], NB);
	    if( res != CUBLAS_STATUS_SUCCESS ) {
		    fprintf(stdout, "%d: cublasGemm error: %d\n", gpu, res );
		    fflush(stdout);
		    return -1;
	    }
	} // end for

	    cublasGetMatrixAsync( NB, NB, sizeof(double_type),
		    d_C[gpu], NB, C+iNB+jNB*N, N,
		   stream[gpu] );
  
#if 0
    	    errno = cudaMemcpy2DAsync(
		C+iNB+jNB*N,
		N*sizeof(double_type),
		d_C[gpu],
		NB*sizeof(double_type),
		NB,
		NB,
		cudaMemcpyDeviceToHost,
		stream[gpu]
		);
#endif
	}
    }

    for( gpu= 0; gpu < ngpu; gpu++ ) {
	errno = cudaStreamSynchronize( stream[gpu] );
	if( errno != cudaSuccess ) {
	    fprintf(stdout, "%d: cudaStreamSynchronize error: %d\n",
		    gpu, errno );
	    fflush(stdout);
	    return -1;
	}
    }
    t1 = get_elapsedtime();

    double tdelta = t1 - t0;
    double gflops = 1.0e-9 * ((2.0 * N * N * N)/(t1-t0));
    printf("# method     size   time      GFlop/s\n");
    printf("DGEMM %6d %10.10f %9.6f\n", (int)N, tdelta, gflops);
    fflush(stdout);
    for( gpu= 0; gpu < ngpu; gpu++ ) {
	fprintf(stdout, "Free gpu %d\n", gpu );
	fflush(stdout);
	cudaSetDevice(gpu);
	cublasDestroy( handle[gpu] );
	cudaStreamDestroy( stream[gpu] );
	cudaFree( d_A[gpu] );
	cudaFree( d_B[gpu] );
	cudaFree( d_C[gpu] );
    }
    if( verif ) {
	    /* Check the solution */
	    info_solution = do_check( A, B, Cinit, C, N );

	    if (info_solution == 0) {
		printf("# TESTING DGEMM .............. PASSED !\n");
	    }
	    else {
		printf("# TESTING DGEMM ... FAILED !\n");
	    }
    	free(Cinit); 
    }

    return 0;
}
