
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

#define CUBLAS_SAFE_THREAD_SYNC( ) do {                                      \
    cublasStatus err = cublasGetError();                                     \
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i.\n",             \
                __FILE__, __LINE__ );                                        \
    } } while(0)

#define CUBLAS_SAFE_CALL_NO_SYNC(call) do {                                 \
    cublasStatus err = call;                                                 \
    if( CUBLAS_STATUS_SUCCESS != err) {                                      \
        fprintf(stderr, "CUBLAS error in file '%s' in line %i.\n",           \
                __FILE__, __LINE__ );                                        \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }while(0)

#define CUBLAS_SAFE_CALL(call)     CUBLAS_SAFE_CALL_NO_SYNC(call)

#else

#define CUDA_SAFE_CALL_NO_SYNC(call)	call
#define CUDA_SAFE_CALL(call) 		call
#define CUDA_SAFE_THREAD_SYNC()

#define CUBLAS_SAFE_CALL_NO_SYNC(call)	call
#define CUBLAS_SAFE_CALL(call) 		call
#define CUBLAS_SAFE_THREAD_SYNC()

#endif /* _DEBUG_ */

#endif /* _CUDA_SAFE_ */
