
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cblas.h>
#include <lapacke.h>

#define CONFIG_USE_DOUBLE 1
#include "test_types.h"


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
    int m, n, k, lda, ldb, ldc;
    double t0, t1;


	if( argc > 1 ){
		m = atoi(argv[1]);
	} else
		m= 512;


	n = k = lda = ldb = ldc = m;

    double_type *A      = (double_type *)malloc(m*n*sizeof(double_type));
    double_type *B      = (double_type *)malloc(n*k*sizeof(double_type));
    double_type *C      = (double_type *)malloc(m*k*sizeof(double_type));

    /* Check if unable to allocate memory */
    if ((!A)||(!B)||(!C)){
        printf("Out of Memory \n ");
        return -2;
    }

    larnv(IONE, ISEED, m*n, A);
    larnv(IONE, ISEED, n*k, B);
    larnv(IONE, ISEED, m*k, C);

      t0 = get_elapsedtime();
    cblas_gemm
    (
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    );
      t1 = get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * m * n * k)/(t1-t0));
    printf("# size   time      GFlop/s\n");
    printf("DGEMM_pt %6d %9.3f %9.3f\n", (int)m, tdelta, gflops);
    fflush(stdout);

    free(A); free(B); free(C);

    return 0;
}

