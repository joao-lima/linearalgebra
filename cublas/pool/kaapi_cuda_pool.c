
#include <stdio.h>
#include <cuda.h>

#if 0
#include "kaapi_impl.h"
#include "kaapi_tasklist.h"

#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_pool.h"

#endif
#include "cuda_runtime.h"

#ifndef CONFIG_USE_FLOAT
#define CONFIG_USE_FLOAT 1
#endif

#include "../test_types.h"

#include "kaapi_cuda_pool.h"

#if KAAPI_CUDA_USE_POOL

/* cuda task body */
//typedef void (*cuda_task_body_t)(void*, CUstream);

static kaapi_cuda_pool_t *pool= NULL;

#ifdef KAAPI_CUDA_POOL_V2
static inline size_t
kaapi_cuda_stream_next( void )
{
    const size_t idx= pool->streamidx;
    pool->streamidx= (idx+1)%KAAPI_CUDA_MAX_STREAMS;
    return idx;
}
#endif

cudaStream_t kaapi_cuda_HtoD_stream( void )
{
#ifdef KAAPI_CUDA_POOL_V2
    return pool->stream[pool->streamidx];
#else
    return pool->stream[KAAPI_CUDA_HTOD_STREAM];
#endif
}

cudaStream_t kaapi_cuda_DtoH_stream( void )
{
#ifdef KAAPI_CUDA_POOL_V2
    return pool->stream[pool->streamidx];
#else
    return pool->stream[KAAPI_CUDA_DTOH_STREAM];
#endif
}

cudaStream_t kaapi_cuda_kernel_stream( void )
{
#ifdef KAAPI_CUDA_POOL_V2
    return pool->stream[pool->streamidx];
#else
    return pool->stream[KAAPI_CUDA_KERNEL_STREAM];
#endif
}

int
kaapi_cuda_htod_is_empty( void )
{
    return (pool->htod_beg == NULL);
}

int
kaapi_cuda_kernel_is_empty( void )
{
    return (pool->kernel_beg == NULL);
}

int
kaapi_cuda_dtoh_is_empty( void )
{
    return (pool->dtoh_beg == NULL);
}

static inline void 
kaapi_cuda_pool_new_event( kaapi_cuda_pool_node_t* node, cudaStream_t stream )
{
    //CUresult res = cudaEventCreate( &node->event, CU_EVENT_DISABLE_TIMING );
    cudaError_t res = cudaEventCreate( &node->event );
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cuEventCreate %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }

    res = cudaEventRecord( node->event, stream );
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cuEventCreate (kernel) %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }

}

static inline kaapi_cuda_pool_node_t* 
kaapi_cuda_pool_node_new( void )
{
    return (kaapi_cuda_pool_node_t*)calloc( 1,
	    sizeof(kaapi_cuda_pool_node_t) );
}

static inline void
kaapi_cuda_pool_node_free( kaapi_cuda_pool_node_t* node )
{
    cuEventDestroy( node->event );
    free( node );
}

void
kaapi_cuda_init( void )
{
    cudaError_t res;
    pool= (kaapi_cuda_pool_t*) calloc( 1, sizeof(kaapi_cuda_pool_t) );
    pool->size= 0;
    pool->htod_beg= NULL;
    pool->htod_end= NULL;
    pool->dtoh_beg= NULL;
    pool->dtoh_end= NULL;
    pool->kernel_beg= NULL;
    pool->kernel_beg= NULL;
    int i;
    for( i= 0; i < KAAPI_CUDA_MAX_STREAMS; i++ ) {
	res = cudaStreamCreate(&pool->stream[i]);
	if (res != CUDA_SUCCESS) {
		fprintf(stdout, "CUDA stream error: %d\n", res );
		fflush(stdout);
	}
    }
#ifdef KAAPI_CUDA_POOL_V2
    pool->streamidx= -1; /* always increment +1 */
#endif
}

void
kaapi_cuda_task_init( void )
{
#ifdef KAAPI_CUDA_POOL_V2
    kaapi_cuda_stream_next(  );
#endif
}

void
kaapi_cuda_task_end( void )
{
}

int
kaapi_cuda_htod_submit( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    kaapi_cuda_pool_node_t* node = kaapi_cuda_pool_node_new();
    node->next = NULL;
    node->last= 0;

    kaapi_cuda_pool_new_event( node, kaapi_cuda_HtoD_stream() );
#if 0
    fprintf(stdout, "[%s] node=%p stream=%ld\n", __FUNCTION__, node,
	  pool->streamidx );
    fflush( stdout );
#endif
    node->state= 0;

    /* insert new event (HtoD) */
    if( NULL == pool->htod_end ) 
	pool->htod_beg= node;
    else
	pool->htod_end->next= node;
    pool->htod_end= node;
#endif

    return 0;
}

int
kaapi_cuda_kernel_submit( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    kaapi_cuda_pool_node_t* node = kaapi_cuda_pool_node_new();
    node->next = NULL;
    node->last= 0;

    kaapi_cuda_pool_new_event( node, kaapi_cuda_kernel_stream() );
#if 0
    fprintf(stdout, "[%s] node=%p stream=%ld\n", __FUNCTION__, 
	    node, pool->streamidx );
    fflush( stdout );
#endif
    node->state= 0;

    /* insert new event */
    if( NULL == pool->kernel_end )
	pool->kernel_beg= node;
    else
	pool->kernel_end->next= node;
    pool->kernel_end= node;
#endif

    return 0;
}

int
kaapi_cuda_dtoh_submit( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    kaapi_cuda_pool_node_t* node = kaapi_cuda_pool_node_new();
    node->next = NULL;
    node->last= 0;

    kaapi_cuda_pool_new_event( node, kaapi_cuda_DtoH_stream() );
#if 0
    fprintf(stdout, "[%s] node=%p stream=%ld\n", __FUNCTION__,
	    node, pool->streamidx );
    fflush( stdout );
#endif
    node->state= 0;

    /* insert new event */
    if( NULL == pool->dtoh_end )
	pool->dtoh_beg= node;
    else
	pool->dtoh_end->next= node;
    pool->dtoh_end= node;
#endif

    return 0;
}

int
kaapi_cuda_htod_end( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    /* checkpoint to the kernel execution */
    kaapi_cuda_pool_node_t *node= pool->htod_end;
    node->last= 1;
    cudaError_t res = cudaStreamWaitEvent( kaapi_cuda_kernel_stream(), node->event, 0);
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cudaStreamWaitEvent %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }
#endif

    return 0;
}

int
kaapi_cuda_kernel_end( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    /* checkpoint to the kernel execution */
    kaapi_cuda_pool_node_t *node= pool->kernel_end;
    node->last= 1;
    cudaError_t res = cudaStreamWaitEvent( kaapi_cuda_DtoH_stream(), node->event, 0);
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cudaStreamWaitEvent %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }
#endif
    return 0;
}

int
kaapi_cuda_dtoh_end( void )
{
#ifndef KAAPI_CUDA_POOL_V2
    /* checkpoint to the kernel execution */
    kaapi_cuda_pool_node_t *node= pool->dtoh_end;
    node->last= 1;
#endif
    return 0;
}

static inline int
_kaapi_cuda_htod_wait( void )
{
    kaapi_cuda_pool_node_t* node;
    node= pool->htod_beg;

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p\n", __FUNCTION__, node );
    fflush( stdout );
#endif
    if( NULL == node )
	return 0;

    switch (node->state) {
	case 0:
	{
#if 0
    fprintf(stdout, "[%s] HtoD node=%p dst=%p src=%p N=%d nb=%d\n",
	    __FUNCTION__, node, node->u.cpy[0], node->u.cpy[1],
	  node->n, node->nb );
    fflush( stdout );
	    kaapi_cuda_pool_new_event( node, kaapi_cuda_HtoD_stream() );
#endif
	    node->state= 1;
	    break;
	}
	case 1:
	{
	    if( CUDA_SUCCESS == cudaEventQuery(node->event) ){
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p DONE\n", __FUNCTION__, node );
    fflush( stdout );
#endif
		pool->htod_beg= node->next;
		if( !kaapi_cuda_kernel_is_empty() && 1 == node->last )
		    pool->kernel_beg->state= 1;
	    }
	    break;
	}
    }

    return 0;
}


static inline int
_kaapi_cuda_kernel_wait( void )
{
    kaapi_cuda_pool_node_t* node;

    node= pool->kernel_beg;
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p\n", __FUNCTION__, node );
    fflush( stdout );
#endif
    if( NULL == node )
	return 0;

    switch (node->state) {
	case 1:
	{
#if 0
    fprintf(stdout, "[%s] GEMM node=%p\n",
	    __FUNCTION__, node);
    fflush( stdout );
#endif
	    node->state= 2;
	    break;
	}
	case 2:
	{
	    if( CUDA_SUCCESS == cudaEventQuery(node->event) ){
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p DONE\n", __FUNCTION__, node );
    fflush( stdout );
#endif
		pool->kernel_beg= node->next;
		if( !kaapi_cuda_dtoh_is_empty() && 1 == node->last )
		    pool->dtoh_beg->state= 1;
	    }
	    break;
	}
    }

    return 0;
}

static inline int
_kaapi_cuda_dtoh_wait( void )
{
    kaapi_cuda_pool_node_t* node;

    node= pool->dtoh_beg;
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p\n", __FUNCTION__, node );
    fflush( stdout );
#endif
    if( NULL == node )
	return 0;

    switch (node->state) {
	case 1:
	{
#if 0
    fprintf(stdout, "[%s] DtoH node=%p dst=%p src=%p N=%d nb=%d\n",
	    __FUNCTION__, node, node->u.cpy[0], node->u.cpy[1],
	  node->n, node->nb );
    fflush( stdout );
	    kaapi_cuda_pool_new_event( node, kaapi_cuda_DtoH_stream() );
#endif
	    node->state= 2;
	    break;
	}
	case 2:
	{
	    if( CUDA_SUCCESS == cudaEventQuery(node->event) ){
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] node=%p DONE\n", __FUNCTION__, node );
    fflush( stdout );
#endif
		pool->dtoh_beg= node->next;
	    }
	    break;
	}
    }

    return 0;
}

int
kaapi_cuda_wait( void )
{
#ifdef KAAPI_CUDA_POOL_V2
    cudaThreadSynchronize();
#else
    do {
	_kaapi_cuda_htod_wait();
	_kaapi_cuda_kernel_wait();
	_kaapi_cuda_dtoh_wait();
#if 0
	node= proc->cuda_proc.pool->htod_beg;
        if( ( NULL != node ) &&
    	        ( CUDA_SUCCESS == cudaEventQuery(node->event) ) ){
	    proc->cuda_proc.pool->htod_beg= node->next;
	    if( node->next == NULL )
		proc->cuda_proc.pool->htod_end= NULL;
	    kaapi_cuda_pool_launch_kernel( proc, node );
        }

	node= proc->cuda_proc.pool->kernel_beg;
        if( ( NULL != node ) &&
    	        ( CUDA_SUCCESS == cudaEventQuery(node->event) ) ){
	    proc->cuda_proc.pool->kernel_beg= node->next;
	    if( node->next == NULL )
		proc->cuda_proc.pool->kernel_end= NULL;
	    kaapi_cuda_mem_sync_params_dtoh( thread, node->td, node->pc );
	    kaapi_cuda_pool_node_free( node );
	}
#endif

    } while( (kaapi_cuda_htod_is_empty()   == 0) ||
	     (kaapi_cuda_kernel_is_empty() == 0) ||
	     (kaapi_cuda_dtoh_is_empty()   == 0) );
#endif

    return 0;
}

#endif /* KAAPI_CUDA_USE_POOL */
