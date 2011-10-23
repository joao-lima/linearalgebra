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

static void create_task_11(starpu_data_handle dataA, unsigned k)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl11;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, k, k);
	task->buffers[0].mode = STARPU_RW;

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_12(starpu_data_handle dataA, unsigned k, unsigned j)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl12;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, k, k); 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = starpu_data_get_sub_data(dataA, 2, j, k); 
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (j == k+1))
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_21(starpu_data_handle dataA, unsigned k, unsigned i)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl21;
	
	/* which sub-data is manipulated ? */
	task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, k, k); 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = starpu_data_get_sub_data(dataA, 2, k, i); 
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

static void create_task_22(starpu_data_handle dataA, unsigned k, unsigned i, unsigned j)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, k, i);
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = starpu_data_get_sub_data(dataA, 2, j, k);
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = starpu_data_get_sub_data(dataA, 2, j, i);
	task->buffers[2].mode = STARPU_RW;

	if (!no_prio &&  (i == k + 1) && (j == k +1) )
		task->priority = STARPU_MAX_PRIO;

	starpu_task_submit(task);
}

/*
 *	code to bootstrap the factorization 
 */

static void dw_codelet_facto_v3(starpu_data_handle dataA, unsigned nblocks)
{
	struct timeval start;
	struct timeval end;

	/* create all the DAG nodes */
	unsigned i,j,k;

	gettimeofday(&start, NULL);

	for (k = 0; k < nblocks; k++)
	{
		create_task_11(dataA, k);
		
		for (i = k+1; i<nblocks; i++)
		{
			create_task_12(dataA, k, i);
			create_task_21(dataA, k, i);
		}

		for (i = k+1; i<nblocks; i++)
		for (j = k+1; j<nblocks; j++)
				create_task_22(dataA, k, i, j);
	}

	/* stall the application until the end of computations */
	starpu_task_wait_for_all();

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stdout, "%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

void STARPU_LU(lu_decomposition)(TYPE *matA, unsigned size, unsigned ld, unsigned nblocks)
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

	dw_codelet_facto_v3(dataA, nblocks);

	/* gather all the data */
	starpu_data_unpartition(dataA, 0);
}
