/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

/*
 * As a convention, in that file, buffers[0] is represented by A,
 * 				  buffers[1] is B ...
 */

/*
 *	Number of flops of Gemm 
 */

#include <starpu.h>
#include "cholesky.h"

struct starpu_perfmodel chol_model_dpotrf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "chol_model_dpotrf"
};

struct starpu_perfmodel chol_model_dsyrk =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "chol_model_dsyrk"
};

struct starpu_perfmodel chol_model_dtrsm =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "chol_model_dtrsm"
};

struct starpu_perfmodel chol_model_dgemm =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "chol_model_dgemm"
};
