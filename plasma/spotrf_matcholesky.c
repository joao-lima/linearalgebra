
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <plasma.h>
#include <cblas.h>
#include <lapacke.h>
#include <plasma_tmg.h>
#include <core_blas.h>

#define CONFIG_USE_FLOAT

#include "test_types.h"

int check_factorization(int, double_type*, double_type*, int, int);

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

    int cores = 2;
    int N     = 1024 ;
    int LDA   = N ;
    int info;
    int info_factorization;
    int i,j;
    int NminusOne = N-1;
    int verif = 0;
    double t0, t1;
    int niter= 1, iter;

    if( argc > 1 ) {
	N = atoi( argv[1] );
	LDA = N;
	NminusOne = N-1;
    }
    if( argc > 2 ) 
	    cores = atoi( argv[2] );

    if( argc > 3 )
      verif = atoi( argv[3] );

    // Cholesky factorization of A 
    double sumt = 0.0;
    double sumgf = 0.0;
    double sumgf2 = 0.0;
    double sd;
    double gflops;
    double gflops_max = 0.0;

    /* formula used by plasma in time_dpotrf.c */
    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (N * (1.0 / 6.0 * N + 0.5 ) * N);
    double fadds = (N * (1.0 / 6.0 * N ) * N);
        

    double_type *A1   = (double_type *)malloc(LDA*N*sizeof(double_type));
    double_type *A2   = (double_type *)malloc(LDA*N*sizeof(double_type));
    double_type *WORK = (double_type *)malloc(2*LDA*sizeof(double_type));
    double_type *D                = (double_type *)malloc(LDA*sizeof(double_type));

    /* Check if unable to allocate memory */
    if ((!A1)||(!A2)){
        printf("Out of Memory \n ");
        exit(0);
    }

    /* Plasma Initialize */
    PLASMA_Init(cores);

    fprintf(stdout, "# PLASMA cores=%d N=%d\n", cores, N);fflush(stdout);

    for( iter= 0; iter < niter; iter++ ){

    /* Initialize A1 and A2 for Symmetric Positive Matrix */
    larnv(IONE, ISEED, LDA, D);
    lagsy(&N, &NminusOne, D, A1, &LDA, ISEED, WORK, &info);
    for ( i = 0; i < N; i++)
        for (  j = 0; j < N; j++)
            A2[LDA*j+i] = A1[LDA*j+i];

    for ( i = 0; i < N; i++){
        A1[LDA*i+i] = A1[LDA*i+i]+ (double_type)N ;
        A2[LDA*i+i] = A1[LDA*i+i];
    }

      t0 = get_elapsedtime();
    /* Plasma routines */
    PLASMA_potrf(PlasmaUpper, N, A2, LDA);
      t1 = get_elapsedtime();

      gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
      if (gflops > gflops_max) gflops_max = gflops;
      
      sumt += t1-t0;
      sumgf += gflops;
      sumgf2 += gflops*gflops;

    if( verif ) {
    /* Check the factorization */
    info_factorization = check_factorization( N, A1, A2, LDA, PlasmaUpper);

    if ((info_factorization != 0)|(info != 0))
       fprintf( stdout, "-- Error in SPOTRF example !\n");
    //else
    //   fprintf( stdout, "-- Run of SPOTRF example successful ! \n");
    }

    }

    gflops = sumgf/niter;

    printf("# size   #threads    time      GFlop/s\n");
    printf("cholesky %6d %5d %9.3f %9.3f\n", (int)N, (int)cores, sumt/niter, gflops);

    free(A1); free(A2); free(WORK); free(D);

    PLASMA_Finalize();

    exit(0);
}


/*------------------------------------------------------------------------
 *  Check the factorization of the matrix A2
 */

int check_factorization(int N, double_type *A1, double_type *A2, int LDA, int uplo)
{
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
    if (uplo == PlasmaUpper){
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

    Rnorm = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, N, Residual, N, work);
    Anorm = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, N, A1, LDA, work);

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
}
