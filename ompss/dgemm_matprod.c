
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cblas.h>
#include <lapacke.h>

#if defined(CONFIG_USE_CUDA)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#endif

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for sLAPACKE_dlarnv() */

static int do_check(
	const double** a,
       	const double** b,
	double** c,
	double** c_old,
	unsigned int nb,
	unsigned int blocsize
	)
{
  double** const tmp = c_old;
  int i, j, k;

  for( i= 0; i < nb; i++ ){
      for( j= 0; j < nb; j++ ){
	  for( k= 0; k < nb; k++ ){
		cblas_dgemm
		(
		    CblasColMajor, CblasNoTrans, CblasNoTrans,
		    blocsize, blocsize, blocsize, 1.0,
		    a[k*nb+i], blocsize,
		    b[j*nb+k], blocsize,
		    1.0, tmp[j*nb+i], blocsize
		);
	  }
      }
  }

  int res = -1;
  int ibloc, jbloc;
  for( ibloc= 0; ibloc < nb; ibloc++ ){
      for( jbloc= 0; jbloc < nb; jbloc++ ){
	  for( i = 0; i < blocsize; i++ ){
	    for( j = 0; j < blocsize; j++ ) {
	      int k = j * blocsize + i;
	      int kbloc = jbloc * nb + ibloc;
	      if( fabsf(c[kbloc][k] - tmp[kbloc][k]) >= 0.01 ) {
		printf("# ERROR invalid %p(%d,%d)(%d,%d) %f != %f\n",
			c[kbloc], ibloc, jbloc, i, j, c[kbloc][k], tmp[kbloc][k] );
		goto on_error;
	      }
	    }
	  }
      }
  }

  res = 0;

 on_error:
  return res;
}
double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

#pragma omp target device(smp)
void dgemm_create_task( 
	const double* const a,
	const double* const  b,
	double* const c,
	const size_t blocsize 
	)
{
    const double alpha = (double) 1.0;
    const double beta = (double) 1.0;
    cblas_dgemm
    (
      CblasColMajor, CblasNoTrans, CblasNoTrans,
      blocsize, blocsize, blocsize,
      alpha,
      a, blocsize,
      b, blocsize,
      beta, c, blocsize
    );
}

int
main( int argc, char **argv )
{
    size_t M = 512, N, K, lda, ldb, ldc;
    size_t blocsize = 256;
    size_t nb;
    size_t i, j, k;
    double t0, t1;
    int verif = 0;
    const double alpha = (double) 1.0;
    const double beta = (double) 1.0;
    double **a, **b, **c, **Cinit;

    if( argc > 1 )
	M = atol(argv[1]);
    if( argc > 2 )
	blocsize = atol(argv[2]);
    if (argc > 3)
	verif = atoi(argv[3]);

    N = K = lda = ldb = ldc = M;
    nb = M / blocsize;

    a      = (double**) malloc( nb * nb * sizeof(double*));
    b      = (double**) malloc( nb * nb * sizeof(double*));
    c      = (double**) malloc( nb * nb * sizeof(double*));
    Cinit  = NULL;
    if ((!a)||(!b)||(!c)){
        printf("Out of Memory \n ");
        abort();
    }
    if( verif ) {
	Cinit  = (double**) malloc( nb * nb * sizeof(double*));
    }

    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
	    a[j*nb+i] = (double*) malloc( blocsize * blocsize * sizeof(double) );
	    b[j*nb+i] = (double*) malloc( blocsize * blocsize * sizeof(double) );
	    c[j*nb+i] = (double*) malloc( blocsize * blocsize * sizeof(double) );
		if( (NULL == a[j*nb+i]) || (NULL == b[j*nb+i]) || (NULL == c[j*nb+i]) ){
		    fprintf(stdout, "ERROR cannot allocate memory\n");
		abort();
	    }
	    LAPACKE_dlarnv( IONE, ISEED, blocsize * blocsize, a[j*nb+i] );
	    LAPACKE_dlarnv( IONE, ISEED, blocsize * blocsize, b[j*nb+i] );
	    LAPACKE_dlarnv( IONE, ISEED, blocsize * blocsize, c[j*nb+i] );

	    if( verif ){
		Cinit[j*nb+i]= (double*) calloc(blocsize * blocsize, sizeof(double));
		if( Cinit[j*nb+i] == 0){
		    fprintf(stdout, "ERROR cannot allocate memory\n");
		    abort();
		}
		memcpy( Cinit[j*nb+i], c[j*nb+i], blocsize * blocsize*sizeof(double) );
	    }
	}
    }
  double* dA;
  double* dB;
  double* dC;

#define A(x,y)	    (a[nb*x+y])
#define B(x,y)	    (b[nb*x+y])
#define C(x,y)	    (c[nb*x+y])
    t0 = get_elapsedtime();
    for( i = 0; i < nb; i++ ){
	for( j = 0; j < nb; j++ ) {
	    for( k= 0; k < nb; k++ ){
	      dA = A(k,i);
	      dB = B(j,k);
	      dC = C(j,i);
#if 0
#pragma omp task input(dA) input(dB) inout(dC)
          dgemm_create_task( dA, dB, dC, blocsize );
#endif
#if 1
//#if defined(CONFIG_USE_CUDA)
	    #pragma omp target device(cuda) copy_deps
	    #pragma omp task inout([blocsize][blocsize] dC) \
		input([blocsize][blocsize] dA) input([blocsize][blocsize] dB)
	      cublasDgemm(
		      nanos_get_cublas_handle(),
		      CUBLAS_OP_N, CUBLAS_OP_N, 
		      blocsize, blocsize, blocsize,
		      &alpha,
		      dA, blocsize,
		      dB, blocsize,
		      &beta, dC, blocsize 
		);
#endif
	    }
	}
    }
#pragma omp taskwait

    t1 = get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * M * N * K)/(t1-t0));
    printf("# size   time      GFlop/s\n");
    printf("DGEMM %lu %lu %.10f %.6f\n", M, blocsize, tdelta, gflops);
    fflush(stdout);

    if( verif ) {
	int res = do_check( (const double**)a, (const double**) b, c, Cinit, nb, blocsize );

	if (res == 0)
		printf("# TESTING DGEMM .............. PASSED !\n");
	    else
		printf("# TESTING DGEMM ... FAILED !\n");
    }

    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
	    free( a[j*nb+i] );
	    free( b[j*nb+i] );
	    free( c[j*nb+i] );
	    if( verif ){
		free( Cinit[j*nb+i] );
	    }
	}
    }
    free(a); free(b); free(c);

    return 0;
}

