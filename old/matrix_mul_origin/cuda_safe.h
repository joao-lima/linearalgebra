
#ifndef _GPU_CUDA_SAFE_H_
#define _GPU_CUDA_SAFE_H_

#include <cuda.h>
#include <stdio.h>

#ifdef _CUDA_DEBUG
#  define CU_SAFE_CALL_NO_SYNC( call ) do {                                  \
    CUresult err = call;                                                     \
    if( CUDA_SUCCESS != err) {                                               \
        printf( "Cuda driver error %d in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
    } } while (0)

#  define CU_SAFE_CALL( call ) do {                                          \
    CU_SAFE_CALL_NO_SYNC(call);                                              \
    CUresult err = cuCtxSynchronize();                                       \
    if( CUDA_SUCCESS != err) {                                               \
        printf( "Cuda driver error %d in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
    } } while (0)
#  define CU_SAFE_CALL_MSG( call,wmsg ) do {                                          \
    CU_SAFE_CALL_NO_SYNC(call);                                              \
    CUresult err = cuCtxSynchronize();                                       \
    if( CUDA_SUCCESS != err) {                                               \
        printf( "Cuda driver error %d in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
        printf( wmsg);   \
    } } while (0)
#else
#  define CU_SAFE_CALL_NO_SYNC( call ) call
#  define CU_SAFE_CALL( call ) call
#  define CU_SAFE_CALL_MSG( call,msg ) call
#endif

#endif /* _GPU_CUDA_SAFE_H_ */
