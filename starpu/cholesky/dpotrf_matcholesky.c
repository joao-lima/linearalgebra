
#define CONFIG_USE_DOUBLE

#include "cholesky.h"

/*------------------------------------------------------------------------
 *  Check the factorization of the matrix A2
 *  from PLASMA (examples/example_dpotrf.c)
 */
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
double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

static struct starpu_codelet cl_dpotrf =
{
	.modes = { STARPU_RW },
	.type = STARPU_SEQ,
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_dpotrf, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_dpotrf, NULL},
#endif
	.nbuffers = 1,
	.model = &chol_model_dpotrf
};

static struct starpu_codelet cl_dtrsm =
{
	.modes = { STARPU_R, STARPU_RW },
	.type = STARPU_SEQ,
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_dtrsm, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_dtrsm, NULL},
#endif
	.nbuffers = 2,
	.model = &chol_model_dtrsm
};

static struct starpu_codelet cl_dsyrk =
{
	.modes = { STARPU_R, STARPU_RW },
	.type = STARPU_SEQ,
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_dsyrk, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_dsyrk, NULL},
#endif
	.nbuffers = 2,
	.model = &chol_model_dsyrk
};

static struct starpu_codelet cl_dgemm =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.type = STARPU_SEQ,
	.max_parallelism = INT_MAX,
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_dgemm, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_dgemm, NULL},
#endif
	.nbuffers = 3,
	.model = &chol_model_dgemm
};

static int cholesky_starpu( starpu_data_handle_t* dataA, unsigned N, unsigned nb )
{
	int ret;

	/* create all the DAG nodes */
	unsigned m, n, k;

	int prio = STARPU_MAX_PRIO;

#define A(x,y)	    (dataA[nb*x+y])
	for (k = 0; k < nb; k++) {
                ret = starpu_insert_task(
			&cl_dpotrf,
			 STARPU_PRIORITY, prio,
			 STARPU_RW, A(k,k),
			 0);
		if (ret == -ENODEV) return 77;

		for (m = k+1; m<nb; m++) {
                        ret = starpu_insert_task(
				&cl_dtrsm,
				 STARPU_PRIORITY, prio,
				 STARPU_R, A(k,k),
				 STARPU_RW, A(k,m),
				 0);
			if (ret == -ENODEV) return 77;
		}

		for (m = k+1; m<nb; m++) {
                        ret = starpu_insert_task(
				&cl_dsyrk,
				 STARPU_PRIORITY, prio,
				 STARPU_R, A(k,m),
				 STARPU_RW, A(m,m),
				 0);
			if (ret == -ENODEV) return 77;

			for (n = k+1; n < nb; n++){
				    ret = starpu_insert_task(
					    &cl_dgemm,
					     STARPU_PRIORITY, prio,
					     STARPU_R, A(k,m),
					     STARPU_R, A(k,n),
					     STARPU_RW, A(n,m),
					     0);
				    if (ret == -ENODEV) return 77;
			}
		}
	}

	starpu_task_wait_for_all();
	return 0;
}

int cholesky(double **a, unsigned n, unsigned blocsize, unsigned nb)
{
	double t0, t1, tdelta;
	unsigned ncpus, ngpus;
	int i, j;
	int ret;

	ncpus= starpu_cpu_worker_get_count();
	ngpus= starpu_cuda_worker_get_count();

	starpu_data_handle_t *dataA = (starpu_data_handle_t*)malloc( nb * nb * sizeof(starpu_data_handle_t));
	for( i= 0; i < nb; i++ ){
	    for( j= 0; j < nb; j++ ){
		starpu_matrix_data_register(&dataA[i*nb+j], 0, (uintptr_t)a[i*nb+j],
		       blocsize, blocsize, blocsize, sizeof(double) );
	    }
	}

      t0 = get_elapsedtime();

	ret = cholesky_starpu( dataA, n, nb );

      t1 = get_elapsedtime();
    tdelta = t1 - t0;

	for( i= 0; i < nb; i++ ){
	    for( j= 0; j < nb; j++ ){
		starpu_data_unregister(dataA[i*nb+j]);
	    }
	}
	free( dataA );

#define FMULS_POTRF(n) ((n) * (((1. / 6.) * (n) + 0.5) * (n) + (1. / 3.)))
#define FADDS_POTRF(n) ((n) * (((1. / 6.) * (n)      ) * (n) - (1. / 6.)))
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
    double gflops = 1e-9 * FLOPS(n) / tdelta;
	fprintf(stdout, "# CPUs GPUs size blocsize time GFlop/s\n");
	fprintf(stdout, "%s %d %d %d %d %10.10f %9.6f\n", "DPOTRF",
			ncpus, ngpus,
		       	n, blocsize, tdelta, gflops);

	fflush(stdout);
	return ret;
}

static void
generate_matrix( double** a, size_t n, size_t blocsize )
{
    size_t i, j;
    size_t ibloc, jbloc, i_idx, j_idx;
    size_t nb = n/ blocsize;

    for (i = 0; i< n; i++) {
	ibloc = i/blocsize;
	i_idx = i%blocsize;
	a[ ibloc*nb + ibloc][i_idx*blocsize+i_idx ] =
	    a[ibloc*nb + ibloc][i_idx*blocsize+i_idx] + 1.*n; 
	for (j = 0; j < i; j++) {
	  jbloc = j/blocsize;
	  j_idx = j%blocsize;
	  a[ibloc*nb+jbloc][i_idx*blocsize+j_idx] =
	      a[jbloc*nb+ibloc][j_idx*blocsize+i_idx];
	}
    }
#if 0
    for( i = 0; i < blocsize; i++ ){
	ibloc = i/blocsize;
	i_idx = i%blocsize;
	for( j = 0; j < blocsize; j++ ) {
	  jbloc = j/blocsize;
	  j_idx = j%blocsize;
	  printf( " %.2f ",
		  a[ibloc*nb+jbloc][i_idx*blocsize+j_idx]
		);
	}
	printf("\n");
    }
#endif
}


int main(int argc, char **argv)
{
    int ret;
    size_t n = 512, blocsize = 256, nb;
    int verif = 0;
    double **a, *Acopy;
    int i, j;

    if( argc > 1 )
	n = atol(argv[1]);
    if( argc > 2 )
	blocsize = atol(argv[2]);
    if (argc > 3)
	verif = atoi(argv[3]);

    nb = n / blocsize;

    ret = starpu_init(NULL);
    if (ret == -ENODEV)
	    exit(77);

    starpu_helper_cublas_init();

    a      = (double**) malloc( nb * nb * sizeof(double*));
    Acopy  = NULL;
    if ((!a)){
	printf("Out of Memory \n ");
	abort();
    }
    if( verif ) {
	Acopy = (double*) malloc( n * n * sizeof(double));
    }

    int IONE=1;
    int ISEED[4] = {0,0,0,1};   /* initial seed for slarnv() */
    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
		starpu_malloc((void **)&a[j*nb+i], blocsize * blocsize * sizeof(double));
		if( NULL == a[j*nb+i] ){
		    fprintf(stdout, "ERROR cannot allocate memory\n");
		abort();
	    }
	    LAPACKE_dlarnv( IONE, ISEED, blocsize * blocsize, a[j*nb+i] );
	}
    }
    generate_matrix( a, n, blocsize );
    if( verif ){
	size_t ibloc, i_idx, jbloc, j_idx;
	for( i = 0; i < blocsize; i++ ){
	    ibloc = i/blocsize;
	    i_idx = i%blocsize;
	    for( j = 0; j < blocsize; j++ ) {
	      jbloc = j/blocsize;
	      j_idx = j%blocsize;
	      Acopy[i*n+j] = a[ibloc*nb+jbloc][i_idx*blocsize+j_idx];
	    }
	}
    }

    ret = cholesky( a, n, blocsize, nb );
    if( verif ) {
	size_t ibloc, i_idx, jbloc, j_idx;
	double *aa = (double*) malloc( n * n * sizeof(double) );
	for( i = 0; i < blocsize; i++ ){
	    ibloc = i/blocsize;
	    i_idx = i%blocsize;
	    for( j = 0; j < blocsize; j++ ) {
	      jbloc = j/blocsize;
	      j_idx = j%blocsize;
	      aa[i*n+j] = a[ibloc*nb+jbloc][i_idx*blocsize+j_idx];
	    }
	}
	check_factorization( n, Acopy, aa, n, CblasLower );
	free( aa );
	free( Acopy );
    }

    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
	    starpu_free( a[j*nb+i] );
	}
    }
    free(a);
    starpu_helper_cublas_shutdown();
    starpu_shutdown();

    return 0;
}


