
#include <cblas.h>
#include <lapacke.h>

#if defined(CONFIG_USE_CUDA)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#endif

void dpotrf_create_task(
	double* const a,
	const size_t blocsize
    )
{
#if 0
    fprintf(stdout, "DPOTRF a=%p\n", 
	    (void*)a );
    fflush(stdout);
#endif
    //clapack_dpotrf( CblasColMajor, CblasLower, blocsize, a, blocsize );
    LAPACKE_dpotrf_work(LAPACK_COL_MAJOR, 'L', blocsize, a, blocsize);
}

void dsyrk_create_task(
	const double* const a,
	double* const c,
	const size_t blocsize
    )
{
    const double alpha = (double) -1.0;
    const double beta = (double) 1.0;

#if 0
    fprintf( stdout, "DSYRK a=%p c=%p\n",
	    (void*)a, (void*)c );
    fflush(stdout);
#endif
    cublasDsyrk_v2(
	  nanos_get_cublas_handle(),
	  CUBLAS_FILL_MODE_LOWER,
	  CUBLAS_OP_N, 
	  blocsize, blocsize,
	  &alpha, a, blocsize,
	  &beta, c, blocsize
    );
}

void dtrsm_create_task(
	const double* const a,
	double* const c,
	const size_t blocsize
    )
{
    const double alpha = (double) 1.0;

#if 0
    fprintf( stdout, "DTRSM a=%p c=%p\n",
	    a, c );
    fflush(stdout);
#endif
    cublasDtrsm_v2 (
	  nanos_get_cublas_handle(),
	  CUBLAS_SIDE_RIGHT,
	  CUBLAS_FILL_MODE_LOWER,
	  CUBLAS_OP_T, 
	  CUBLAS_DIAG_NON_UNIT,
	  blocsize, blocsize,
	  &alpha, a, blocsize,
	  c, blocsize
    );
}

void dgemm_create_task( 
	const double* const a,
	const double* const  b,
	double* const c,
	const size_t blocsize 
	)
{
    const double alpha = (double) 1.0;
    const double beta = (double) 1.0;
#if 0
    fprintf( stdout, "DGEMM a=%p b=%p c=%p\n",
	    (void*)a, (void*)b, (void*)c );
    fflush(stdout);
#endif
    cublasDgemm_v2(
	  nanos_get_cublas_handle(),
	  CUBLAS_OP_N, CUBLAS_OP_N, 
	  blocsize, blocsize, blocsize,
	  &alpha,
	  a, blocsize,
	  b, blocsize,
	  &beta, c, blocsize 
    );
}

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for sLAPACKE_dlarnv() */

int check_factorization(int N, double *A1, double *A2, int LDA, int uplo)
{
    double Anorm, Rnorm;
    double alpha;
    int info_factorization;
    int i,j;
    double eps;

    eps = LAPACKE_dlamch_work('e');

    double *Residual = (double *)malloc(N*N*sizeof(double));
    double *L1       = (double *)malloc(N*N*sizeof(double));
    double *L2       = (double *)malloc(N*N*sizeof(double));
    double *work              = (double *)malloc(N*sizeof(double));

    memset((void*)L1, 0, N*N*sizeof(double));
    memset((void*)L2, 0, N*N*sizeof(double));

    alpha= 1.0;

    LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,' ', N, N, A1, LDA, Residual, N);

    /* Dealing with L'L or U'U  */
    if (uplo == CblasUpper){
        LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L1, N);
        LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L2, N);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    else{
        LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L1, N);
        LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L2, N);
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }

    /* Compute the Residual || A -L'L|| */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
           Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];

    Rnorm = LAPACKE_dlange_work(LAPACK_COL_MAJOR, 'I', N, N, Residual, N, work);
    Anorm = LAPACKE_dlange_work(LAPACK_COL_MAJOR, 'I', N, N, A1, LDA, work);

    printf("# ============\n");
    printf("# Checking the Cholesky Factorization \n");
    printf("# -- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));

    if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
        printf("# ERROR -- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else{
        printf("# OK -- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    free(Residual); free(L1); free(L2); free(work);

    return info_factorization;
}

