
#ifndef KAAPI_CUDA_POOL_H
#define KAAPI_CUDA_POOL_H

#define KAAPI_CUDA_USE_POOL 1

//#define KAAPI_VERBOSE	1

#define KAAPI_CUDA_MAX_STREAMS		8
#define KAAPI_CUDA_HTOD_STREAM		0
#define KAAPI_CUDA_KERNEL_STREAM        1
#define KAAPI_CUDA_DTOH_STREAM		2
#define KAAPI_CUDA_DTOD_STREAM		3

//#define KAAPI_CUDA_POOL_V2	    1

//#include "kaapi.h"
//#include "kaapi_cuda_proc.h"

#include "cuda.h"
#include "cublas_v2.h"

typedef struct kaapi_cuda_pool_node {
    cudaEvent_t event;
    int last;
    int state;
    struct kaapi_cuda_pool_node* next;
} kaapi_cuda_pool_node_t;

typedef struct kaapi_cuda_pool {
    unsigned int size;
#ifdef KAAPI_CUDA_POOL_V2
    size_t streamidx;
#endif
    cudaStream_t stream[KAAPI_CUDA_MAX_STREAMS];
    kaapi_cuda_pool_node_t* htod_beg;
    kaapi_cuda_pool_node_t* htod_end;
    kaapi_cuda_pool_node_t* kernel_beg;
    kaapi_cuda_pool_node_t* kernel_end;
    kaapi_cuda_pool_node_t* dtoh_beg;
    kaapi_cuda_pool_node_t* dtoh_end;
} kaapi_cuda_pool_t;

#if 0
void kaapi_cuda_pool_submit_HtoD( );
void kaapi_cuda_pool_submit_DtoH( );
void kaapi_cuda_pool_submit_kernel( );
#endif

void kaapi_cuda_init( void  );

void kaapi_cuda_task_init( void );

void kaapi_cuda_task_end( void );

int kaapi_cuda_htod_submit( void );

int kaapi_cuda_htod_end( void );

int kaapi_cuda_kernel_submit( void );

int kaapi_cuda_kernel_end( void );

int kaapi_cuda_dtoh_submit( void );

int kaapi_cuda_dtoh_end( void );

int kaapi_cuda_wait( void );

int kaapi_cuda_htod_is_empty( void );

int kaapi_cuda_kernel_is_empty( void );

int kaapi_cuda_dtoh_is_empty( void );

cudaStream_t kaapi_cuda_HtoD_stream( void );

cudaStream_t kaapi_cuda_DtoH_stream( void );

cudaStream_t kaapi_cuda_kernel_stream( void );

#endif /* KAAPI_CUDA_POOL_H */
