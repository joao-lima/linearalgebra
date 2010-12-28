
#ifndef _CUDA_SAFE_
#define _CUDA_SAFE_

#ifdef _DEBUG

#  define CUDA_SAFE_THREAD_SYNC( ) do {                                      \
    cudaError err = cudaThreadSynchronize();                                 \
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    } } while(0)

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }while(0)

#define CUDA_SAFE_CALL(call)     CUDA_SAFE_CALL_NO_SYNC(call)

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

#define CUDA_SAFE_CALL_NO_SYNC(call)	call
#define CUDA_SAFE_CALL(call) 		call
#define CUDA_SAFE_THREAD_SYNC()

#  define CU_SAFE_CALL_NO_SYNC( call ) call
#  define CU_SAFE_CALL( call ) call
#  define CU_SAFE_CALL_MSG( call,msg ) call
#endif /* _DEBUG_ */

#endif /* _CUDA_SAFE_ */
