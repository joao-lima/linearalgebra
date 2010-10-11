
#ifndef _CUDA_DEFS_
#define _CUDA_DEFS_

// Thread block size
#define BLOCK_SIZE 256
#ifndef DEVICE
#define DEVICE	0
#endif

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

#else

#define CUDA_SAFE_CALL_NO_SYNC(call)	call
#define CUDA_SAFE_CALL(call) 		call
#define CUDA_SAFE_THREAD_SYNC()

#endif /* _DEBUG_ */

#endif /* _CUDA_DEFS_ */