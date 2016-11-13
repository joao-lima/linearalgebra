/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#ifndef __DW_CHOLESKY_H__
#define __DW_CHOLESKY_H__

#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#include <cblas.h>
#include <clapack.h>
#include <lapacke.h>
#include <starpu.h>

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)
#define NMAXBLOCKS	32

#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG12(k,i)	((starpu_tag_t)(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

#define BLOCKSIZE	(size/nblocks)

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

void chol_cpu_codelet_update_dpotrf(void **, void *);
void chol_cpu_codelet_update_dsyrk(void **, void *);
void chol_cpu_codelet_update_dtrsm(void **, void *);
void chol_cpu_codelet_update_dgemm(void **, void *);

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_dpotrf(void *descr[], void *_args);
void chol_cublas_codelet_update_dsyrk(void *descr[], void *_args);
void chol_cublas_codelet_update_dtrsm(void *descr[], void *_args);
void chol_cublas_codelet_update_dgemm(void *descr[], void *_args);
#endif

extern struct starpu_perfmodel chol_model_dpotrf;
extern struct starpu_perfmodel chol_model_dsyrk;
extern struct starpu_perfmodel chol_model_dtrsm;
extern struct starpu_perfmodel chol_model_dgemm;

#endif /* __DW_CHOLESKY_H__ */
