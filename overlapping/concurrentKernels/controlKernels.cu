
#include <stdio.h>

#include "add1_kernel.cu"

int main(int argc, char **argv)
{
    int cuda_device = 0;
    unsigned int mem_size = (1 << MAX_MEM);
    unsigned int ntasks = NTASKS;
    unsigned int itasks = 0; // n of ready tasks
    float *h_data[NTASKS], *d_data[NTASKS];
    float elapsed_time= 0;

    cuda_device = 0;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL( cudaGetDevice(&cuda_device));	

    CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (deviceProp.concurrentKernels == 0 ))
        printf("> GPU does not support concurrent kernel execution, kernel runs will be serialized\n");

    for( int i= 0; i < ntasks; i++ ) {
	CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i], mem_size) ); 
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data[i], mem_size) );
	for( int j= 0; j < (mem_size/sizeof(float)); j++ )
		h_data[i][j]= 1.0f;
    }
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
	
    cudaStream_t stream_HtoD, stream_DtoH, stream_k;
    CUDA_SAFE_CALL( cudaStreamCreate(&stream_HtoD) );
    CUDA_SAFE_CALL( cudaStreamCreate(&stream_DtoH) );
    CUDA_SAFE_CALL( cudaStreamCreate(&stream_k) );

    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    cudaEventRecord(start_event, 0);

    unsigned int i_HtoD= 0, i_DtoH= 0, i_k= 0;

  CUDA_SAFE_CALL( cudaMemcpyAsync( d_data[i_HtoD],
	h_data[i_HtoD], mem_size, cudaMemcpyHostToDevice, stream_HtoD ));
	   i_HtoD++;
    while( itasks < ntasks ) {

	if( i_k > 0 ) {
		CUDA_SAFE_CALL( cudaStreamSynchronize( stream_k ) );
		if( i_DtoH < ntasks ) {
//		  fprintf(stdout,"DtoH for k=%d\n",i_DtoH);fflush(stdout);
		CUDA_SAFE_CALL( cudaMemcpyAsync( h_data[i_DtoH],
			d_data[i_DtoH], mem_size,
			cudaMemcpyDeviceToHost, stream_DtoH ) );
		i_DtoH++;
		itasks++;
		}
	}

	CUDA_SAFE_CALL( cudaStreamSynchronize( stream_HtoD ) );
	if( i_k < ntasks ) {
//	  fprintf(stdout,"kernel k=%d\n",i_k);fflush(stdout);
	   add1<<<GRID_SIZE,BLOCK_SIZE,0,stream_k>>>(d_data[i_k], (mem_size/sizeof(float)) );
       i_k++;
	}

       if( i_HtoD < ntasks ) {
//	  fprintf(stdout,"HtoD k=%d\n",i_HtoD);fflush(stdout);
	  CUDA_SAFE_CALL( cudaMemcpyAsync( d_data[i_HtoD],
		h_data[i_HtoD], mem_size,
		cudaMemcpyHostToDevice, stream_HtoD ));
	   i_HtoD++;
       }

    }

//    fprintf(stdout,"sync\n");fflush(stdout);
    CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time,
		    start_event, stop_event) );
    
    printf("Measured time for sample = %.4f\n", elapsed_time);

    for( int i= 0; i < ntasks; i++ )
	    if( check( h_data[i], mem_size/sizeof(float), 2) )
		    fprintf(stdout, "ERROR at task %d\n", i ); fflush(stdout);
    
    for( int i= 0; i < ntasks; i++ ) {
	    cudaFreeHost(h_data[i]);
	    cudaFree(d_data[i]);
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaThreadExit();
}
