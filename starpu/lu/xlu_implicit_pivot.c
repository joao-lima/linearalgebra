/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
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

#include "xlu.h"
#include "xlu_kernels.h"

static unsigned no_prio = 0;

/*
 *	Construct the DAG
 */

static void create_task_pivot(starpu_data_handle *dataAp, unsigned nblocks,
					struct piv_s *piv_description,
					unsigned k, unsigned i,
					starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl_pivot;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, nblocks, k, i);
	task->buffers[0].mode = STARPU_RW;

	task->cl_arg = &piv_description[k];

	/* this is an important task */
	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_11_pivot(starpu_data_handle *dataAp, unsigned nblocks,
					unsigned k, struct piv_s *piv_description,
					starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl11_pivot;

	task->cl_arg = &piv_description[k];

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, nblocks, k, k);
	task->buffers[0].mode = STARPU_RW;

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_12(starpu_data_handle *dataAp, unsigned nblocks, unsigned k, unsigned j,
		starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = starpu_task_create();
	
	task->cl = &cl12;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, nblocks, k, k);
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, nblocks, j, k);
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (j == k+1))
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_21(starpu_data_handle *dataAp, unsigned nblocks, unsigned k, unsigned i,
				starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl21;
	
	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, nblocks, k, k); 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, nblocks, k, i); 
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_22(starpu_data_handle *dataAp, unsigned nblocks, unsigned k, unsigned i, unsigned j,
				starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, nblocks, k, i);
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, nblocks, j, k);
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = get_block(dataAp, nblocks, j, i);
	task->buffers[2].mode = STARPU_RW;

	if (!no_prio &&  (i == k + 1) && (j == k +1) )
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

/*
 *	code to bootstrap the factorization 
 */

static double dw_codelet_facto_pivot(starpu_data_handle *dataAp,
					struct piv_s *piv_description,
					unsigned nblocks,
					starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	/* create all the DAG nodes */
	unsigned i,j,k;
	for (k = 0; k < nblocks; k++)
	{
		create_task_11_pivot(dataAp, nblocks, k, piv_description, get_block);

		for (i = 0; i < nblocks; i++)
		{
			if (i != k)
				create_task_pivot(dataAp, nblocks, piv_description, k, i, get_block);
		}
	
		for (i = k+1; i<nblocks; i++)
		{
			create_task_12(dataAp, nblocks, k, i, get_block);
			create_task_21(dataAp, nblocks, k, i, get_block);
		}

		for (i = k+1; i<nblocks; i++)
		for (j = k+1; j<nblocks; j++)
			create_task_22(dataAp, nblocks, k, i, j, get_block);
	}

	/* stall the application until the end of computations */
	starpu_task_wait_for_all();

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	return timing;
}

starpu_data_handle get_block_with_striding(starpu_data_handle *dataAp,
			unsigned nblocks __attribute__((unused)), unsigned j, unsigned i)
{
	/* we use filters */
	return starpu_data_get_sub_data(*dataAp, 2, j, i);
}


void STARPU_LU(lu_decomposition_pivot)(TYPE *matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, 0, (uintptr_t)matA, ld, size, size, sizeof(TYPE));

	struct starpu_data_filter f;
		f.filter_func = starpu_vertical_block_filter_func;
		f.nchildren = nblocks;
		f.get_nchildren = NULL;
		f.get_child_ops = NULL;

	struct starpu_data_filter f2;
		f2.filter_func = starpu_block_filter_func;
		f2.nchildren = nblocks;
		f2.get_nchildren = NULL;
		f2.get_child_ops = NULL;

	starpu_data_map_filters(dataA, 2, &f, &f2);

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / nblocks);
		piv_description[block].last = (block + 1) * (size / nblocks);
	}

	double timing;
	timing = dw_codelet_facto_pivot(&dataA, piv_description, nblocks, get_block_with_striding);

	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stderr, "%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));

	/* gather all the data */
	starpu_data_unpartition(dataA, 0);
}


starpu_data_handle get_block_with_no_striding(starpu_data_handle *dataAp, unsigned nblocks, unsigned j, unsigned i)
{
	/* dataAp is an array of data handle */
	return dataAp[i+j*nblocks];
}

void STARPU_LU(lu_decomposition_pivot_no_stride)(TYPE **matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle *dataAp = malloc(nblocks*nblocks*sizeof(starpu_data_handle));

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	unsigned bi, bj;
	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		starpu_matrix_data_register(&dataAp[bi+nblocks*bj], 0,
			(uintptr_t)matA[bi+nblocks*bj], size/nblocks,
			size/nblocks, size/nblocks, sizeof(TYPE));
	}

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / nblocks);
		piv_description[block].last = (block + 1) * (size / nblocks);
	}

	double timing;
	timing = dw_codelet_facto_pivot(dataAp, piv_description, nblocks, get_block_with_no_striding);

	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stderr, "%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataAp[0])*nblocks;
	double flop = (2.0f*n*n*n)/3.0f;
	FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));

	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		starpu_data_unregister(dataAp[bi+nblocks*bj]);
	}
}
