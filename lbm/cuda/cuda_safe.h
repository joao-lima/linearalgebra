
#ifndef _CUDA_SAFE_
#define _CUDA_SAFE_

#ifdef _DEBUG
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
#endif /* _DEBUG_ */

#endif /* _CUDA_SAFE_ */
