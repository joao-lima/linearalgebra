
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "cblas.h"
#include "lapacke.h"

#include "magma.h"
#include "magmablas.h"
//#include "magma_lapack.h"

#define CONFIG_USE_DOUBLE 1
#include "test_types.h"

int check_factorization(int, double_type*, double_type*, int, char);

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

    int N     = 1024 ;
    int info= 0;
    int info_factorization;
    int NminusOne = N-1;
    int verif = 0;
    double t0, t1;
    int niter= 1, iter;
    int lwork;
    double_type tmp[1];

    if( argc > 1 ) {
	N = atoi( argv[1] );
	NminusOne = N-1;
    }

    if( argc > 2 )
      verif = atoi( argv[2] );

    // Cholesky factorization of A 
    double sumt = 0.0;
    double sumgf = 0.0;
    double sumgf2 = 0.0;
    double gflops;
    double gflops_max = 0.0;

#if 0
    /* formula used by plasma in time_dpotrf.c */
    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (N * (1.0 / 6.0 * N + 0.5 ) * N);
    double fadds = (N * (1.0 / 6.0 * N ) * N);
#endif
        

    double_type *A   = (double_type *)malloc(N*N*sizeof(double_type));
    double_type *tau   = (double_type *)malloc(N*sizeof(double_type));
    lwork= -1; /* calculate lwork into tmp */
    magma_dgeqrf( N, N, A, N, tau, tmp, lwork, &info );
    lwork = tmp[0];
    double_type *work   = (double_type *)malloc(lwork*sizeof(double_type));
    double_type *A2= NULL;
    if( verif )
	A2 = (double_type *)malloc(N*N*sizeof(double_type));

    /* Check if unable to allocate memory */
    if (!A || !tau || !work ){
        printf("Out of Memory \n ");
        exit(0);
    }

    cudaSetDevice( 0 );
    cudaFree(0);
    for( iter= 0; iter < niter; iter++ ){
	/* Initialize A1 and A2 for Symmetric Positive Matrix */
	larnv( IONE, ISEED, N, A );

	t0 = get_elapsedtime();
	magma_dgeqrf( N, N, A, N, tau, work, lwork, &info );
	t1 = get_elapsedtime();

	if( info < 0 ){
	    fprintf(stdout,"magma_dgeqrf ERROR: %d\n", info);
	    return info;
	}
	//      gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
#define FLOPS(m,n) (      FMULS_GEQRF(m,n) +      FADDS_GEQRF(m,n) )
	gflops = 1e-9 * FLOPS(N,N) / (t1-t0);
	if (gflops > gflops_max) gflops_max = gflops;

	sumt += t1-t0;
	sumgf += gflops;
	sumgf2 += gflops*gflops;

	if( verif ) {
	    /* Check the factorization */
//	    info_factorization = check_factorization( N, A, A2, N, uplo);

	    if ((info_factorization != 0)|(info != 0))
	    fprintf( stdout, "-- Error in DGEQRF example !\n");
	}

    }

    gflops = sumgf/niter;

    printf("# method     size   time      GFlop/s\n");
    printf("DGEQRF %6d %10.10f %9.6f\n", (int)N, sumt/niter, gflops);
    fflush(stdout);

    cudaDeviceReset();
    free(A);
    if( verif )
	free( A2 );

    return 0;
}


/*------------------------------------------------------------------------
 *  Check the factorization of the matrix A2
 */

int check_factorization(int N, double_type *A1, double_type *A2, int LDA, char uplo)
{
#if 0
    double_type Anorm, Rnorm;
    double_type alpha;
    int info_factorization;
    int i,j;
    double_type eps;

    eps = lamch('e');

    double_type *Residual = (double_type *)malloc(N*N*sizeof(double_type));
    double_type *L1       = (double_type *)malloc(N*N*sizeof(double_type));
    double_type *L2       = (double_type *)malloc(N*N*sizeof(double_type));
    double_type *work              = (double_type *)malloc(N*sizeof(double_type));

    memset((void*)L1, 0, N*N*sizeof(double_type));
    memset((void*)L2, 0, N*N*sizeof(double_type));

    alpha= 1.0;

    lacpy(LAPACK_COL_MAJOR,' ', N, N, A1, LDA, Residual, N);

    /* Dealing with L'L or U'U  */
    if (uplo == MagmaUpper){
        lacpy(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L1, N);
        lacpy(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L2, N);
        cblas_trmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    else{
        lacpy(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L1, N);
        lacpy(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L2, N);
        cblas_trmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }

    /* Compute the Residual || A -L'L|| */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
           Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];

    char infnorm[]= "";
    Rnorm = lange(LAPACK_COL_MAJOR, infnorm[0], N, N, Residual, N, work);
    Anorm = lange(LAPACK_COL_MAJOR, infnorm[0], N, N, A1, LDA, work);

    //printf("============\n");
    //printf("Checking the Cholesky Factorization \n");
    fprintf( stdout, "# ||L'L-A||_oo/(||A||_oo.N.eps) = %e\n",
		    Rnorm/(Anorm*N*eps));
    fflush(stdout);

    if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
        printf("# ERRO Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else{
        //printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    free(Residual); free(L1); free(L2); free(work);

    return info_factorization;
#endif
    return 0;
}