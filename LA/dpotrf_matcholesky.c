
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

static void
generate_matrix(double_type* A, int m)
{
	int i, j;
  // 
  for (i = 0; i< m; ++i)
  {
    for (j = 0; j< m; ++j)
      A[i*m+j] = 1.0 / (1.0+i+j);
    A[i*m+i] = m*1.0; 
  }
}

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
    double t0, t1;


	if( argc > 1 ){
		n = atoi(argv[1]);
	} else
		n= 512;


    double_type *A      = (double_type *)malloc(n*n*sizeof(double_type));

    /* Check if unable to allocate memory */
    if ( !A ){
        printf("Out of Memory \n ");
        return -2;
    }

    generate_matrix( A, n );

      t0 = get_elapsedtime();
	clapack_potrf( CblasRowMajor, CblasLower, n, A, n );
      t1 = get_elapsedtime();
    double tdelta = t1 - t0;

    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (n * (1.0 / 6.0 * n + 0.5 ) * n);
    double fadds = (n * (1.0 / 6.0 * n ) * n);
    double gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / tdelta;
        
    printf("# size   time      GFlop/s\n");
    printf("DPOTRF %6d %9.3f %9.3f\n", (int)n, tdelta, gflops);
    fflush(stdout);

    free(A);

    return 0;
}

