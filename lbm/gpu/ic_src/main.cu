#include "lb.h"

///////////////////////////////////////////////
int main(int argc, char **argv) {

	//Parameters

	//Iteration counter
	int time;

	//Execution Time
	double execution_time;

	//Input structure
	s_properties *properties;
	double *vel, *v;
	cudaMalloc((void**)&vel,sizeof(double));
	v = (double*) calloc(1,sizeof(double));

	//Lattice structure
	s_lattice *lattice;

	//startup information message
	/* printf("Lattice Boltzmann Method\n");
	printf("Claudio Schepke\n");
	printf("Instituto de Informática - UFRGS\n");
	printf("Date: 2006, January 09\n\n"); */

	//Checking arguments
	if (argc != 6) {
		fprintf(stderr, "Usage: %s [file_parametrs] [file_obstacles] [file_output] [grid_dimension] [block_dimension]\n\n", argv[0]);
		exit(1);
	}

	//Begin initialization
	
	//Read parameter file
	//properties->t_max
	//properties->density
	//properties->accel
	//properties->omega
	//properties->r_rey
	printf("Reading parameters...\n");
	properties = (s_properties*) read_parametrs(argv[1]);


	/*************** BEGINNING OF CUDA CODE ***************/
	s_properties *GPUproperties;
	cudaMalloc((void**)&GPUproperties, sizeof(s_properties));
	cudaMemcpy(GPUproperties, properties, sizeof(s_properties), cudaMemcpyHostToDevice);
	/*************** END OF CUDA CODE ***************/


	//read obstacle file
	//<x> <y> <n directions> <number of obstacles> 
	//x-,and y-coordinates of any obstacles
	//wall boundaries are also defined here by adding single obstacles
	printf("Reading obstacles...\n");
	lattice = (s_lattice*) read_obstacles(argv[2]);

        printf("Initializing lattice density...\n");
        init_density(lattice, properties->density);
	
	printf("Allocating GPU memory for the lattice structure...\n");
	s_lattice *GPUlattice;
	cudaMalloc((void**)&GPUlattice, sizeof(s_lattice));
	cudaMemcpy(GPUlattice, lattice, sizeof(s_lattice), cudaMemcpyHostToDevice);

	dim3 one_block(1, 1, 1);
	dim3 one_thread(1, 1, 1);

	//GPUinit_density<<< one_block, one_thread >>>(GPUlattice, properties->density);
	double *n_sum, *sum;
	cudaMalloc((void **)&sum, sizeof(double));
	n_sum = (double*) calloc(1, sizeof(double));

	dim3 blocks_of_the_grid(atoi(argv[4]), atoi(argv[4]), 1);
	dim3 threads_per_block(atoi(argv[5]), atoi(argv[5]), 1);
        //dim3 blocks_of_the_grid(16, 16, 1);
        //dim3 threads_per_block(16, 16, 1);

	execution_time = crono();

	//Begin of the main loop
	for (time = 0; time < properties->t_max; time++) {

	  if (!(time%(properties->t_max/1))) {
	    GPUcheck_density<<< one_block, one_thread >>>(GPUlattice, sum);
	    cudaMemcpy(n_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
	    printf("Iteration number = %d\n", time);
            printf("Integral density = %lf\n", *n_sum); 
	  }
		
	//	GPUmain_loop<<< blocks_of_the_grid, threads_per_block >>>(GPUlattice, GPUproperties, time);
	//	cudaThreadSynchronize();

	  GPUredistribute<<< one_block, one_thread >>>(GPUlattice, properties->accel, properties->density);
	  cudaThreadSynchronize();

	  GPUpropagate<<< blocks_of_the_grid, threads_per_block >>>(GPUlattice);
	  cudaThreadSynchronize();

	  GPUbounceback<<< blocks_of_the_grid, threads_per_block >>>(GPUlattice);
	  cudaThreadSynchronize();

	  GPUrelaxation<<< blocks_of_the_grid, threads_per_block >>>(GPUlattice, properties->density, properties->omega);
	  cudaThreadSynchronize();

	  GPUcalc_velocity<<< one_block, one_thread >>>(GPUlattice, time, vel);
	  cudaThreadSynchronize();

	  cudaMemcpy(v, vel, sizeof(double), cudaMemcpyDeviceToHost);
printf("%d %lf\n", time, *v);
	  if (time%500 == 0) {
	  //  printf("%d %lf\n", time, *v);
	    FILE *c = fopen("convergence9.out", "a");
	    fprintf(c, "%d %lf\n", time, *v);
	    fclose(c);
	  }
	}

	execution_time = crono() - execution_time;
	printf("Total execution time: %f\n", execution_time);
		
	GPUcomp_rey<<< one_block, one_thread >>>(GPUlattice, GPUproperties, time);


	/*************** BEGINNING OF CUDA CODE ***************/
	// copy properties structure to cpu
	cudaMemcpy(properties, GPUproperties, sizeof(s_properties), cudaMemcpyDeviceToHost);
	cudaMemcpy(lattice, GPUlattice, sizeof(s_lattice), cudaMemcpyDeviceToHost);
	/*************** END OF CUDA CODE ***************/


	write_results(argv[3], lattice, properties, time, execution_time, atoi(argv[4]), atoi(argv[5]));

	//printf("End of the execution\n\n");
	return 1;
}
