
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <plasma.h>
#include <cblas.h>
#include <lapacke.h>
#include <core_blas.h>

#define CONFIG_USE_FLOAT 1

#include "test_types.h"

int check_solution(int, int , double_type *, int, double_type *, double_type *, int);

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for slarnv() */

double_type get_LU_error(int M, int N, 
		    double_type *A,  int lda, 
		    double_type *LU, int *IPIV)
{
    int min_mn = N;
    int ione   = 1;
    int i, j;
   double_type  alpha = 1;
   double_type beta  = 0;
 //   float *L, *U;
    double_type work[1], matnorm, residual;
                       
    double_type *L = (double_type *)malloc(N*N*(sizeof(double_type)));
    double_type *U = (double_type *)malloc(N*N*(sizeof(double_type)));
    memset( L, 0, N*N*sizeof(double_type) );
    memset( U, 0, N*N*sizeof(double_type) );

    laswp( LAPACK_COL_MAJOR, N, A, lda, ione, N, IPIV, ione);
    lacpy( LAPACK_COL_MAJOR, 'l', N, N, LU, lda, L, N     );
    lacpy( LAPACK_COL_MAJOR, 'u', N, N, LU, lda, U, N );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = 1.0;
    
    matnorm = lange( LAPACK_COL_MAJOR, 'f', N, N, A, lda, work);

    cblas_gemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, N,
                  alpha, L, M, U, N, beta, LU, lda );

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = LU[i+j*lda] - A[i+j*lda] ;
	}
    }
    residual = lange( LAPACK_COL_MAJOR, 'f', M, N, LU, lda, work);

    free(L); free(U);

    return residual / (matnorm * N);
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

    int cores = 2;
    int N     = 512;
    int NRHS  = 5;
    int LDB   = 10;
    int info;
    int info_solution;
    int i,j;
    int verif = 0;
    double t0, t1;
    int niter= 1, iter;

    if( argc > 1 ) {
	N = atoi( argv[1] );
    }
    if( argc > 2 ) 
	    cores = atoi( argv[2] );

    if( argc > 3 )
      verif = atoi( argv[3] );

//    int LDA   = N;
//    int LDAxN = LDA*N;
//    int LDBxNRHS = LDB*NRHS;
    double_type *A1 = (double_type *)malloc(N*N*sizeof(double_type));
    double_type *A2;
//    double_type *A2 = (double_type *)malloc(LDA*N*(sizeof*A2));
//    double_type *B1 = (double_type *)malloc(LDB*NRHS*(sizeof*B1));
//    double_type *B2 = (double_type *)malloc(LDB*NRHS*(sizeof*B2));
    double_type *L;
    int *IPIV;

    if( verif ){
    	A2 = (double_type *)malloc(N*N*sizeof(double_type));
	if( (!A2) ){
		printf("Out of Memory \n ");
		exit(0);
	    }
    }
    /* Check if unable to allocate memory */
    if ( (!A1)  ) {
        printf("Out of Memory \n ");
        exit(0);
    }

    /*Plasma Initialize*/
    PLASMA_Init(cores);

    fprintf(stdout, "# PLASMA cores=%d N=%d\n", cores, N);fflush(stdout);

    /* Allocate L and IPIV */
    info = PLASMA_Alloc_Workspace_getrf_incpiv(N, N, &L, &IPIV);

    double ggflops = 0;
    double gtime = 0;
    //printf("-- PLASMA is initialized to run on %d cores. \n",cores);
    for( iter= 0; iter < niter; iter++ ){

    /* Initialize A1 and A2 Matrix */
    larnv(IONE, ISEED, N*N, A1);
    if( verif )
	    lacpy( LAPACK_COL_MAJOR, ' ', N, N, A1, N, A2, N);

      t0 = get_elapsedtime();
    /* LU factorization of the matrix A */
    info = PLASMA_getrf(N, N, A1, N, L, IPIV);
      t1 = get_elapsedtime();

      double fp_per_mul = 1;
      double fp_per_add = 1;
      double fmuls = (N * (1.0 / 3.0 * N )      * N);
      double fadds = (N * (1.0 / 3.0 * N - 0.5) * N);
      double gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
      gtime += t1-t0;
      ggflops += gflops;

      if( verif ){ 
	double_type error = get_LU_error( N, N, A2, N, A1, IPIV);
	fprintf(stdout, "# LU error %e\n", error );
      }

    }

    printf("# size   #threads    time      GFlop/s\n");
    printf("LU %6d %5d %9.3f %9.3f\n", (int)N, (int)cores, gtime/niter,
		    ggflops/niter);
    free(A1); free(IPIV); free(L);

    PLASMA_Finalize();

    exit(0);
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution of the linear system
 */
#if 0
int check_solution(int N, int NRHS, double_type *A1, int LDA, double_type *B1, double_type *B2, int LDB)
{
    int info_solution;
    double_type Rnorm, Anorm, Xnorm, Bnorm;
    double_type alpha, beta;
    double_type *work = (double_type *)malloc(N*sizeof(double_type));
    double_type eps;

    eps = LAPACKE_slamch_work('e');

    alpha = 1.0;
    beta  = -1.0;

    Xnorm = LAPACKE_slange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B2, LDB, work);
    Anorm = LAPACKE_slange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, N, A1, LDA, work);
    Bnorm = LAPACKE_slange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B1, LDB, work);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, NRHS, N, (alpha), A1, LDA, B2, LDB, (beta), B1, LDB);
    Rnorm = LAPACKE_slange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B1, LDB, work);

    printf("# Checking the Residual of the solution \n");
    printf("# ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));

    if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*N*eps)) || (Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 10.0) ){
        printf("# ERROR -- The solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
//        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }

    free(work);

    return info_solution;
}
#endif
