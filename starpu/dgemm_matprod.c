
#include <stdio.h>
#include <sys/time.h>

#include "cblas.h"
#include "clapack.h"
#include "lapacke.h"

#include <starpu_config.h>
#include <starpu.h>

#ifdef STARPU_USE_CUDA
#include <starpu_cuda.h>
#ifdef STARPU_HAVE_MAGMA
#include "magma.h"
#endif
#endif

static int do_check
	(const double* a, const double* b, double* c,
 	double* cfinal, unsigned int n)
{
  double* const tmp = (double*)calloc(n * n, sizeof(double));
  if (tmp == NULL) return -1;
  double alpha = (double) 1.0;
  double beta = (double) 1.0;

  LAPACKE_dlacpy( LAPACK_COL_MAJOR, ' ', n, n, c, n, tmp, n );

  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
	a, n, b, n, beta, tmp, n );

  unsigned int i, j, k;
  int res = -1;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      k = i * n + j;
      if (fabsf(cfinal[k] - tmp[k]) >= 0.01)
      {
	printf("ERROR invalid @%p,%u,%u %f != %f\n", c, i, j, cfinal[k], tmp[k]);
	goto on_error;
      }
    }
  }

  res = 0;

 on_error:
  free(tmp);

  return res;
}
#define PERTURBATE(a)	(a)

static double cpu_gemm_task_dgemm_cost(struct starpu_task *task, enum starpu_perf_archtype arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/8.0760);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cpu_chol_task_dgemm_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double cuda_gemm_task_dgemm_cost(struct starpu_task *task, enum starpu_perf_archtype arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/76.30666);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cuda_chol_task_dgemm_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

struct starpu_perfmodel gemm_model_dgemm =
{
	.per_arch =
	{
		[STARPU_CPU_DEFAULT][0] = { .cost_function = cpu_gemm_task_dgemm_cost },
		[STARPU_CUDA_DEFAULT][0] = { .cost_function = cuda_gemm_task_dgemm_cost }
	},
	.type = STARPU_HISTORY_BASED,
	.symbol = "gemm_model_dgemm"
};

static inline void gemm_common_cpu_codelet_update_dgemm(
	void *descr[], int s, __attribute__((unused)) void *_args)
{
	double *a		= (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	double *b		= (double *)STARPU_MATRIX_GET_PTR(descr[1]);
	double *c	 	= (double *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned m = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned n = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned k = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned lda = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldb = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldc = STARPU_MATRIX_GET_LD(descr[2]);

#if 0
	fprintf(stdout, "[%s] DGEMM (s=%d) m=%d n=%d k=%d lda=%d ldb=%d ldc=%d A=%p B=%p C=%p\n",
		__FUNCTION__,
		s,
		m, n, k,
		lda, ldb, ldc,
		a, b, c
		);
	fflush(stdout);
#endif

	if (s == 0) {
		/* Sequential CPU kernel */
		cblas_dgemm(
			CblasColMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, 
			1.0f, a, lda,
			b, ldb,
			1.0f, c, ldc
			);
	} else {
		/* CUDA kernel */
#ifdef STARPU_USE_CUDA
		cublasDgemm('n', 'n', m, n, k,
				1.0f, a, lda, b, ldb, 
				1.0f, c, ldc 
				 );
		cudaStreamSynchronize(starpu_cuda_get_local_stream());
#endif
	}
}

void gemm_cpu_codelet_update_dgemm(void *descr[], void *_args)
{
	gemm_common_cpu_codelet_update_dgemm(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void gemm_cublas_codelet_update_dgemm(void *descr[], void *_args)
{
	gemm_common_cpu_codelet_update_dgemm(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

/*
 *	Some useful functions
 */

double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}


static struct starpu_codelet cl_dgemm =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.type = STARPU_SEQ,
	.max_parallelism = INT_MAX,
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {gemm_cpu_codelet_update_dgemm, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {gemm_cublas_codelet_update_dgemm, NULL},
#endif
	.nbuffers = 3,
	.model = &gemm_model_dgemm
};

static int
create_task_dgemm(
	starpu_data_handle_t A,
	starpu_data_handle_t B,
	starpu_data_handle_t C,
       	unsigned i, unsigned j, unsigned k )
{
    int ret;
    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_dgemm;

    task->handles[0] = starpu_data_get_sub_data( A, 2, k, j );
    task->handles[1] = starpu_data_get_sub_data( B, 2, i, k );
    task->handles[2] = starpu_data_get_sub_data( C, 2, i, j );
    task->priority = STARPU_MAX_PRIO;
    ret = starpu_task_submit( task );

    return ret;
}

static int dgemm_starpu(
	starpu_data_handle_t A,
	starpu_data_handle_t B,
	starpu_data_handle_t C,
	unsigned n, unsigned nblocks
	)
{
	int ret;

	/* create all the DAG nodes */
	unsigned i, j, k;

	for (j = 0; j < nblocks; j++) {
	    for( i = 0; i < nblocks; i++ ){
		for( k= 0; k < nblocks; k++ ){
		    ret = create_task_dgemm( A, B, C, i, j, k );
		    if (ret == -ENODEV) return 77;
		}
	    }
	}

	starpu_task_wait_for_all();
	return 0;
}

int dgemm(
	double* const A,
	double* const B, 
	double* C, 
	unsigned n, unsigned nblocks
    )
{
	double t0, t1, tdelta;
	unsigned ncpus, ngpus;
	int ret;

	ncpus= starpu_cpu_worker_get_count();
	ngpus= starpu_cuda_worker_get_count();

	starpu_data_handle_t A_handle, B_handle, C_handle;
	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register( &A_handle, 0, (uintptr_t)A, n, n, n, sizeof(double) );
	starpu_matrix_data_register( &B_handle, 0, (uintptr_t)B, n, n, n, sizeof(double) );
	starpu_matrix_data_register( &C_handle, 0, (uintptr_t)C, n, n, n, sizeof(double) );

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vertical_block_filter_func,
		.nchildren = nblocks
	};

	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_block_filter_func,
		.nchildren = nblocks
	};

	starpu_data_map_filters( A_handle, 2, &f, &f2 );
	starpu_data_map_filters( B_handle, 2, &f, &f2 );
	starpu_data_map_filters( C_handle, 2, &f, &f2 );

      t0 = get_elapsedtime();

	ret = dgemm_starpu( A_handle, B_handle, C_handle, n, nblocks );

      t1 = get_elapsedtime();
    tdelta = t1 - t0;

	starpu_data_unpartition( A_handle, 0 );
	starpu_data_unpartition( B_handle, 0 );
	starpu_data_unpartition( C_handle, 0 );
	starpu_data_unregister( A_handle );
	starpu_data_unregister( B_handle );
	starpu_data_unregister( C_handle );

	double gflops = 1.0e-9 * ((2.0 * n * n * n)/(t1-t0));
	fprintf(stdout, "# CPUs GPUs size blocsize time GFlop/s\n");
	fprintf(stdout, "%s %d %d %d %d %10.10f %9.6f\n", "DGEMM",
			ncpus, ngpus,
		       	n, (n/nblocks),tdelta, gflops
		);

	fflush(stdout);
	return ret;
}

int main(int argc, char **argv)
{
     	int ret;
	int n = 256, nblocks = 1, verif = 0;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	starpu_helper_cublas_init();

	if (argc > 1)
	    n = atoi(argv[1]);

	if (argc > 2)
	    nblocks = atoi(argv[2]);

	if (argc > 3)
	    verif = atoi(argv[3]);

	double *A, *B, *C, *Cinit;
	starpu_malloc( (void **)&A, n*n*sizeof(double) );
	starpu_malloc( (void **)&B, n*n*sizeof(double) );
	starpu_malloc( (void **)&C, n*n*sizeof(double) );

	int IONE=1;
	int ISEED[4] = {0,0,0,1};   /* initial seed for slarnv() */
	LAPACKE_dlarnv( IONE, ISEED, n*n, A );
	LAPACKE_dlarnv( IONE, ISEED, n*n, B );
	LAPACKE_dlarnv( IONE, ISEED, n*n, C );
	if( verif ) {
	    Cinit  = (double*)malloc(n*n*sizeof(double));
	    memcpy( Cinit, C, n*n*sizeof(double) );
	}

	ret = dgemm( A, B, C, n, nblocks );

	if( verif ) {
#if 0
	    int i, j;
	    for( i= 0; i < n; i++) {
		for( j= 0; j< n; j++ ){
		    printf( " %.2f ", C[i*n+j] );
		}
		printf("\n");
	    }
#endif
	    int res = do_check( A, B, Cinit, C, n );

	    if (res == 0)
		    printf("# TESTING DGEMM .............. PASSED !\n");
		else
		    printf("# TESTING DGEMM ... FAILED !\n");

	    free(Cinit); 
	}

	starpu_free( A );
	starpu_free( B );
	starpu_free( C );
	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
