
#ifndef _CUDA_DEFS_
#define _CUDA_DEFS_

#include <math.h>

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

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for equality
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
inline int compareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        fprintf( stdout, "ERROR, reference l2-norm is 0\n");
#endif
        return 0;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
#ifdef _DEBUG
    if( ! (error < epsilon)) 
        fprintf( stdout, "ERROR, l2-norm error %f is greater than epsilon %f\n",
		 error, epsilon );
#endif

    return ( error < epsilon );
}

#endif /* _CUDA_DEFS_ */
