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
#include "blas/blas.h"
#ifdef STARPU_USE_CUDA
#include <starpu_cuda.h>
#endif

/*
 *   U22 
 */

static inline void chol_common_cpu_codelet_update_u22(void *descr[], int s, __attribute__((unused)) void *_args)
{
	/* printf("22\n"); */
	double_type *left 	= (double_type *)STARPU_MATRIX_GET_PTR(descr[0]);
	double_type *right 	= (double_type *)STARPU_MATRIX_GET_PTR(descr[1]);
	double_type *center 	= (double_type *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned dx = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned dy = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned dz = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ld22 = STARPU_MATRIX_GET_LD(descr[2]);

	if (s == 0)
	{
		int worker_size = starpu_combined_worker_get_size();

		if (worker_size == 1)
		{
			/* Sequential CPU kernel */
			GEMM("N", "T", dy, dx, dz, -1.0f, left, ld21, 
				right, ld12, 1.0f, center, ld22);
		}
		else {
			/* Parallel CPU kernel */
			int rank = starpu_combined_worker_get_rank();

			int block_size = (dx + worker_size - 1)/worker_size;
			int new_dx = STARPU_MIN(dx, block_size*(rank+1)) - block_size*rank;
			
			double_type *new_left = &left[block_size*rank];
			double_type *new_center = &center[block_size*rank];

			GEMM("N", "T", dy, new_dx, dz, -1.0f, new_left, ld21, 
				right, ld12, 1.0f, new_center, ld22);
		}
	}
	else
	{
		/* CUDA kernel */
#ifdef STARPU_USE_CUDA
		cublasGemm('n', 't', dy, dx, dz, 
				-1.0f, left, ld21, right, ld12, 
				 1.0f, center, ld22);
		cudaStreamSynchronize(starpu_cuda_get_local_stream());
#endif

	}
}

void chol_cpu_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

/* 
 * U21
 */

static inline void chol_common_codelet_update_u21(void *descr[], int s, __attribute__((unused)) void *_args)
{
/*	printf("21\n"); */
	double_type *sub11;
	double_type *sub21;

	sub11 = (double_type *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub21 = (double_type *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx21 = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned ny21 = STARPU_MATRIX_GET_NX(descr[1]);

	switch (s) {
		case 0:
			TRSM("R", "L", "T", "N", nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			cublasTrsm('R', 'L', 'T', 'N', nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_u21(void *descr[], void *_args)
{
	 chol_common_codelet_update_u21(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u21(void *descr[], void *_args)
{
	chol_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void chol_common_codelet_update_u11(void *descr[], int s, __attribute__((unused)) void *_args) 
{
/*	printf("11\n"); */
	double_type *sub11;

	sub11 = (double_type *)STARPU_MATRIX_GET_PTR(descr[0]); 

	unsigned nx = STARPU_MATRIX_GET_NY(descr[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned z;

	switch (s) {
		case 0:

			/*
			 *	- alpha 11 <- lambda 11 = sqrt(alpha11)
			 *	- alpha 21 <- l 21	= alpha 21 / lambda 11
			 *	- A22 <- A22 - l21 trans(l21)
			 */

			for (z = 0; z < nx; z++)
			{
				double_type lambda11;
				lambda11 = sqrt(sub11[z+z*ld]);
				sub11[z+z*ld] = lambda11;

				STARPU_ASSERT(lambda11 != 0.0f);
		
				SCAL(nx - z - 1, 1.0f/lambda11, &sub11[(z+1)+z*ld], 1);
		
				SYR("L", nx - z - 1, -1.0f, 
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			{
			double_type *lambda11;
			cudaHostAlloc((void **)&lambda11, sizeof(double_type), 0);

			for (z = 0; z < nx; z++)
			{

				cudaMemcpyAsync(lambda11, &sub11[z+z*ld], sizeof(double_type), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
				cudaStreamSynchronize(starpu_cuda_get_local_stream());

				STARPU_ASSERT(*lambda11 != 0.0f);
				
				*lambda11 = sqrt(*lambda11);

/*				cublasSetVector(1, sizeof(double_type), lambda11, sizeof(double_type), &sub11[z+z*ld], sizeof(double_type)); */
				cudaMemcpyAsync(&sub11[z+z*ld], lambda11, sizeof(double_type), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());

				cublasScal(nx - z - 1, 1.0f/(*lambda11), &sub11[(z+1)+z*ld], 1);

				cublasSyr('U', nx - z - 1, -1.0f,
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}

			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			cudaFreeHost(lambda11);
			}
		

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}


void chol_cpu_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 1, _args);
}
#endif/* STARPU_USE_CUDA */
