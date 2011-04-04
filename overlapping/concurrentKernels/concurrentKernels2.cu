
#include <stdio.h>

#define	NSTREAMS	4
#include "add1_kernel.cu"

int main(int argc, char **argv)
{
    int cuda_device = 0;
    unsigned int mem_size = (1 << MAX_MEM);
    unsigned int nstreams = NSTREAMS;
    unsigned int ntasks = NTASKS;
    unsigned int nevents = ntasks * 2;
    float *h_data[NTASKS], *d_data[NTASKS];
    float elapsed_time= 0;

    cuda_device = 0;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL( cudaGetDevice(&cuda_device));	

    CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (deviceProp.concurrentKernels == 0 ))
        printf("> GPU does not support concurrent kernel execution, kernel runs will be serialized\n");

    //printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount); 

    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    cudaEvent_t *events = (cudaEvent_t*) malloc(nevents * sizeof(cudaEvent_t));
    for(int i = 0; i < nstreams; i++)
	CUDA_SAFE_CALL( cudaStreamCreate(&(streams[i])) );
    for(int i = 0; i < nevents; i++)
        CUDA_SAFE_CALL( cudaEventCreateWithFlags(&(events[i]), cudaEventDisableTiming) );

    for( int i= 0; i < ntasks; i++ ) {
	CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i], mem_size) ); 
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data[i], mem_size) );
	for( int j= 0; j < (mem_size/sizeof(float)); j++ )
		h_data[i][j]= 1.0f;
    }
    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

    cudaStream_t DtoH, HtoD, k_stream;
    CUDA_SAFE_CALL( cudaStreamCreate(&DtoH) );
    CUDA_SAFE_CALL( cudaStreamCreate(&HtoD) );
    CUDA_SAFE_CALL( cudaStreamCreate(&k_stream) );
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    cudaEventRecord(start_event, 0);

    unsigned int istream=0;
    // queue nkernels in separate streams and record when they are done
    for( int i=0; i < ntasks; ++i) {
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_data[i], h_data[i], mem_size,
			cudaMemcpyHostToDevice, streams[istream] ));
        add1<<<GRID_SIZE,BLOCK_SIZE,0,streams[istream]>>>(d_data[i], (mem_size/sizeof(float)) );
    //fprintf(stdout,"HtoD\n");fflush(stdout);
     istream = (istream+1)%nstreams;
    }
#if 1
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    for( int i=0; i < ntasks; ++i) {
	CUDA_SAFE_CALL( cudaMemcpyAsync( h_data[i], d_data[i], mem_size,
			cudaMemcpyDeviceToHost, DtoH ) );
    //fprintf(stdout,"DtoH\n");fflush(stdout);
    }
#endif
    fprintf(stdout,"sync\n");fflush(stdout);

    // in this sample we just wait until the GPU is done
    CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time,
		    start_event, stop_event) );
    
    printf("Measured time for sample = %.4f\n", elapsed_time);

    for( int i= 0; i < ntasks; i++ )
	    if( check( h_data[i], mem_size/sizeof(float), 2) )
		    fprintf(stdout, "ERROR at task %d\n", i ); fflush(stdout);
    
    // release resources
    for(int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);
  cudaStreamDestroy(DtoH);
  cudaStreamDestroy(HtoD);
  cudaStreamDestroy(k_stream);

    for(int i = 0; i < nevents; i++)
		cudaEventDestroy(events[i]);
    for( int i= 0; i < ntasks; i++ ) {
	    cudaFreeHost(h_data[i]);
	    cudaFree(d_data[i]);
    }

    free(streams);
    free(events);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaThreadExit();
}
