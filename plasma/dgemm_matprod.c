
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <plasma.h>
#include <cblas.h>
#include <lapacke.h>
#include <core_blas.h>

//#define CONFIG_USE_FLOAT 1
#define CONFIG_USE_DOUBLE  1

#include "test_types.h"

static int check_solution(PLASMA_enum transA, PLASMA_enum transB, int M, int N, int K,
                          double_type alpha, double_type *A, int LDA,
                          double_type *B, int LDB,
                          double_type beta, double_type *Cref, double_type *Cplasma, int LDC);

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
    int M     = 512;
    int N     = M;
    int K     = M;
    int LDA   = M;
    int LDB   = M;
    int LDC   = M;
    int cores= 1, verif= 0;
    int transa, transb;
    double t0, t1;

    transa = transb = PlasmaNoTrans;

	if( argc > 1 ){
		M = N = K = LDA = LDB = LDC = atoi(argv[1]);
	}
    if( argc > 2 ) 
	    cores = atoi( argv[2] );

    if( argc > 3 )
      verif = atoi( argv[3] );


    double_type eps;
    int info_solution;
    int i, j, ta, tb;
    int LDAxK = LDA*max(M,K);
    int LDBxN = LDB*max(K,N);
    int LDCxN = LDC*N;

    double_type *A      = (double_type *)malloc(LDAxK*sizeof(double_type));
    double_type *B      = (double_type *)malloc(LDBxN*sizeof(double_type));
    double_type *C      = (double_type *)malloc(LDCxN*sizeof(double_type));
	double_type *Cinit;
    if( verif )
		Cinit  = (double_type *)malloc(LDCxN*sizeof(double_type));

    /* Check if unable to allocate memory */
    if ( (!A) || (!B) || (!C) ){
        printf("Out of Memory \n ");
        return -2;
    }

    /* Plasma Initialize */
    PLASMA_Init(cores);

    eps = lamch('e');

    /* Initialize A, B, C */
    larnv(IONE, ISEED, LDAxK, A);
    larnv(IONE, ISEED, LDBxN, B);
    larnv(IONE, ISEED, LDCxN, C);

    if( verif )
    	lacpy( LAPACK_COL_MAJOR, ' ', N, N, C, N, Cinit, N );


      t0 = get_elapsedtime();
    /* PLASMA SGEMM */
    PLASMA_gemm(transa, transb, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
      t1 = get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * M * M * M)/(t1-t0));
    printf("# size   #threads    time      GFlop/s\n");
    printf("DGEMM %6d %5d %10.10f %9.6f\n", (int)N, (int)cores, tdelta, gflops);
    fflush(stdout);

    if( verif ) {
	    /* Check the solution */
	    info_solution = check_solution(transa, transb, M, N, K, 
			   alpha, A, LDA, B, LDB, beta, Cinit, C, LDC);

	    if (info_solution == 0) {
		printf("# TESTING SGEMM .............. PASSED !\n");
	    }
	    else {
		printf("# TESTING SGEMM ... FAILED !\n");
	    }
    free(Cinit);
    }
    free(A); free(B); free(C);

    PLASMA_Finalize();

    return 0;
}

/*--------------------------------------------------------------
 * Check the solution
 */

static int check_solution(PLASMA_enum transA, PLASMA_enum transB, int M, int N, int K,
                          double_type alpha, double_type *A, int LDA,
                          double_type *B, int LDB,
                          double_type beta, double_type *Cref, double_type *Cplasma, int LDC)
{
    int info_solution;
    double_type Anorm, Bnorm, Cinitnorm, Cplasmanorm, Clapacknorm, Rnorm, result;
    double_type eps;
    double_type beta_const;

    double_type *work = (double_type *)malloc(max(K,max(M, N))* sizeof(double_type));
    int Am, An, Bm, Bn;

    beta_const  = -1.0;

    if (transA == PlasmaNoTrans) {
        Am = M; An = K;
    } else {
        Am = K; An = M;
    }
    if (transB == PlasmaNoTrans) {
        Bm = K; Bn = N;
    } else {
        Bm = N; Bn = K;
    }

    Anorm       = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), Am, An, A,       LDA, work);
    Bnorm       = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), Bm, Bn, B,       LDB, work);
    Cinitnorm   = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), M,  N,  Cref,    LDC, work);
    Cplasmanorm = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), M,  N,  Cplasma, LDC, work);

    cblas_gemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, M, N, K, 
                (alpha), A, LDA, B, LDB, (beta), Cref, LDC);

    Clapacknorm = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), M, N, Cref, LDC, work);

    cblas_axpy(LDC * N, (beta_const), Cplasma, 1, Cref, 1);

    Rnorm = lange(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), M, N, Cref, LDC, work);

    eps = lamch('e');

    printf("Rnorm %e, Anorm %e, Bnorm %e, Cinitnorm %e, Cplasmanorm %e, Clapacknorm %e\n", 
           Rnorm, Anorm, Bnorm, Cinitnorm, Cplasmanorm, Clapacknorm);

    result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * N * eps);
    printf("============\n");
    printf("Checking the norm of the difference against reference SGEMM \n");
    printf("-- ||Cplasma - Clapack||_oo/((||A||_oo+||B||_oo+||C||_oo).N.eps) = %e \n", 
           result);

    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
         printf("-- The solution is suspicious ! \n");
         info_solution = 1;
    }
    else {
         printf("-- The solution is CORRECT ! \n");
         info_solution= 0 ;
    }

    free(work);

    return info_solution;
}
