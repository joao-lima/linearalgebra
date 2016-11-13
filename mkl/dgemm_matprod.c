
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//#include <cblas.h>
#include "mkl_cblas.h"
#include "mkl_lapack.h"
//#include <lapacke.h>

//#define CONFIG_USE_DOUBLE 1
//#include "test_types.h"


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
    double alpha = (double) 1.0;
    double beta = (double) 1.0;
    int m, n, k, lda, ldb, ldc;
    double t0, t1;


	if( argc > 1 ){
		m = atoi(argv[1]);
	} else
		m= 512;


	n = k = lda = ldb = ldc = m;

    double *A      = (double *)malloc(m*n*sizeof(double));
    double *B      = (double *)malloc(n*k*sizeof(double));
    double *C      = (double *)malloc(m*k*sizeof(double));

    /* Check if unable to allocate memory */
    if ((!A)||(!B)||(!C)){
        printf("Out of Memory \n ");
        return -2;
    }

    int size = m*n;
    dlarnv(&IONE, ISEED, &size, A);
    size = n*k;
    dlarnv(&IONE, ISEED, &size, B);
    size = m*k;
    dlarnv(&IONE, ISEED, &size, C);

      t0 = get_elapsedtime();
    cblas_dgemm
    (
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    );
      t1 = get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * m * n * k)/(t1-t0));
    printf("# size   time      GFlop/s\n");
    printf("DGEMM %6d %10.10f %9.6f\n", (int)m, tdelta, gflops);
    fflush(stdout);

    free(A); free(B); free(C);

    return 0;
}

