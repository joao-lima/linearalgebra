
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cblas.h>
#include <clapack.h>
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
    int n;
    double t0, t1, *tau;


	if( argc > 1 ){
		n = atoi(argv[1]);
	} else
		n= 512;


    double_type *A      = (double_type *)malloc(n*n*sizeof(double_type));
    tau = (double_type*) calloc( n, sizeof(double_type) );

    /* Check if unable to allocate memory */
    if ( !A || !tau ){
        printf("Out of Memory \n ");
        return -2;
    }

    larnv(IONE, ISEED, n*n, A);

      t0 = get_elapsedtime();
	clapack_dgeqrf( CblasRowMajor, n, n, A, n, tau );
      t1 = get_elapsedtime();
    double tdelta = t1 - t0;

#if 0
    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (n * (1.0 / 6.0 * n + 0.5 ) * n);
    double fadds = (n * (1.0 / 6.0 * n ) * n);
    double gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / tdelta;
#endif
#define FLOPS(m,n) (      FMULS_GEQRF(m,n) +      FADDS_GEQRF(m,n) )
    double gflops = 1e-9 * FLOPS(n,n) / tdelta;
        
    printf("# size   time      GFlop/s\n");
    printf("DGEQRF %6d %10.10f %9.6f\n", (int)n, tdelta, gflops);
    fflush(stdout);

    free(A);
    free(tau);

    return 0;
}

