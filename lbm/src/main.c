#include "lb.h"

///////////////////////////////////////////////
int main(int argc, char **argv) {
	
	//Parameters

	//Iteration counter
	int time;

	//Execution Time
	double execution_time;

	//Average velocity
	double vel;

        double tdelta;
	//Input structure
	s_properties *properties;

	//Lattice structure
	s_lattice *lattice;

	//startup information message
	/* printf("Lattice Boltzmann Method\n");
	printf("Claudio Schepke\n");
	printf("Instituto de Informática - UFRGS\n");
	printf("Date: 2006, January 09\n\n"); */

	//Checking arguments
	if (argc != 4) {
		fprintf(stderr, "Usage: %s [file_configuration] [file_colision] [file_plot]\n\n", argv[0]);
		exit(1);
	}
	
	//Begin initialization
	
	//Read parameter file
	//properties->t_max
	//properties->density
	//properties->accel
	//properties->omega
	//properties->r_rey
	properties = (s_properties*) read_parametrs(argv[1]);

	//read obstacle file
	//<x> <y> <n directions> <number of obstacles> 
	//x-,and y-coordinates of any obstacles
	//wall boundaries are also defined here by adding single obstacles
	lattice = (s_lattice*) read_obstacles(argv[2]);
	
	
	init_density(lattice, properties->density);

	execution_time = crono();
        struct timeval t1, t2;
        gettimeofday( &t1, 0 );

	//Begin of the main loop
	for (time = 0; time < properties->t_max; time++) {
//	for (time = 0; time < 2; time++) {

		 if (!(time%(properties->t_max/1))) {
			check_density(lattice, time);
		}
		
		redistribute(lattice, properties->accel, properties->density);

		propagate(lattice);

		bounceback(lattice);

		relaxation(lattice, properties->density, properties->omega);

		vel = calc_velocity(lattice, time);
		printf("%d %f\n", time, vel);
	}
        gettimeofday( &t2, 0 );
        tdelta = (t2.tv_sec-t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
      
	execution_time = crono() - execution_time;
		
	comp_rey(lattice, properties, time, execution_time);

	write_results(argv[3], lattice, properties->density);

	printf("time(s): %f\n\n",tdelta);
	return 1;
}
