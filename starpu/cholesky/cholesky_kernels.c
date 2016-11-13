/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_config.h>

#include "cholesky.h"

#include "cblas.h"
#include "clapack.h"

#ifdef STARPU_USE_CUDA
#include <starpu_cuda.h>
#ifdef STARPU_HAVE_MAGMA
#include "magma.h"
//#include "magma_lapack.h"
#endif
#endif

/*
 *   U22 
 */

static inline void chol_common_cpu_codelet_update_dgemm(
	void *descr[], int s, __attribute__((unused)) void *_args)
{
	/* printf("22\n"); */
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
			CblasColMajor, CblasNoTrans, CblasTrans,
			m, n, k, 
			-1.0f, a, lda,
			b, ldb,
			1.0f, c, ldc
			);
	} else {
		/* CUDA kernel */
#ifdef STARPU_USE_CUDA
		cublasDgemm('n', 't', m, n, k,
				-1.0f, a, lda, b, ldb, 
				 1.0f, c, ldc 
				 );
		cudaStreamSynchronize(starpu_cuda_get_local_stream());
#endif
	}
}

void chol_cpu_codelet_update_dgemm(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_dgemm(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_dgemm(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_dgemm(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

/* 
 * U21
 */

static inline void chol_common_codelet_update_dtrsm(void *descr[], int s, __attribute__((unused)) void *_args)
{
/*	printf("21\n"); */
	double *a;
	double *c;

	a = (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	c = (double *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned lda = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldc = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned n = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned k = STARPU_MATRIX_GET_NY(descr[0]);

#if 0
	fprintf(stdout, "[%s] DTRSM (s=%d) n=%d k=%d lda=%d ldc=%d A=%p C=%p\n",
		__FUNCTION__,
		s,
		n, k,
		lda, ldc,
		(void*)a, (void*)c
		);
	fflush(stdout);
#endif

	switch (s) {
		case 0:
		    cblas_dtrsm(
			    CblasColMajor, CblasRight, CblasLower,
			    CblasTrans, CblasNonUnit,
			    n, k, 1.0f, a, lda, c, ldc
			    );
		    break;
#ifdef STARPU_USE_CUDA
		case 1:
			cublasDtrsm('R', 'L', 'T', 'N', n, k, 1.0f, a, ldc, c, ldc);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_dtrsm(void *descr[], void *_args)
{
	 chol_common_codelet_update_dtrsm(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_dtrsm(void *descr[], void *_args)
{
	chol_common_codelet_update_dtrsm(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void chol_common_codelet_update_dpotrf(void *descr[], int s, __attribute__((unused)) void *_args) 
{
/*	printf("11\n"); */
	double *a;

	a = (double *)STARPU_MATRIX_GET_PTR(descr[0]); 

	unsigned n = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned lda = STARPU_MATRIX_GET_LD(descr[0]);

#if 0
	fprintf(stdout, "[%s] DPOTRF (s=%d) n=%d lda=%d A=%p\n",
		__FUNCTION__,
		s,
		n, lda, (void*)a );
	fflush(stdout);
#endif

	switch (s) {
		case 0:
			clapack_dpotrf(
				CblasColMajor,
				CblasLower,
				n, a, lda
			    );
			break;
#ifdef STARPU_USE_CUDA
		case 1:
#ifdef STARPU_HAVE_MAGMA
			{
			int ret;
			int info;
			ret = magma_dpotrf_gpu('L', n, a, lda, &info);
			if (ret != MAGMA_SUCCESS)
			{
				fprintf(stderr, "Error in Magma: %d\n", ret);
				STARPU_ABORT();
			}
			cudaError_t cures = cudaThreadSynchronize();
			STARPU_ASSERT(!cures);
			}
#endif

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}


void chol_cpu_codelet_update_dpotrf(void *descr[], void *_args)
{
	chol_common_codelet_update_dpotrf(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_dpotrf(void *descr[], void *_args)
{
	chol_common_codelet_update_dpotrf(descr, 1, _args);
}
#endif/* STARPU_USE_CUDA */

static inline void chol_common_codelet_update_dsyrk(void *descr[], int s, __attribute__((unused)) void *_args)
{
/*	printf("21\n"); */
	double *A;
	double *C;

	A = (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	C = (double *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned lda = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldc = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned n = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned k = STARPU_MATRIX_GET_NX(descr[0]);

#if 0
	fprintf(stdout, "[%s] DSYRK (s=%d) n=%d k=%d lda=%d ldc=%d A=%p C=%p\n",
		__FUNCTION__,
		s,
		n, k,
		lda, ldc,
		(void*)A, (void*)C
		);
	fflush(stdout);
#endif

	switch (s) {
		case 0:
		    cblas_dsyrk(
			    CblasColMajor, CblasLower, CblasNoTrans,
			    n, k,
			    -1.0f, A, lda,
			    1.0f, C, ldc
			    );
		    break;
#ifdef STARPU_USE_CUDA
		case 1:
			cublasDsyrk(
			    'L', 'N',
			    n, k,
			    -1.0f, A, lda,
			    1.0f, C, ldc
			);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_dsyrk(void *descr[], void *_args)
{
	chol_common_codelet_update_dsyrk(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_dsyrk(void *descr[], void *_args)
{
	chol_common_codelet_update_dsyrk(descr, 1, _args);
}
#endif/* STARPU_USE_CUDA */

