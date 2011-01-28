
#include "lb.h"
#include <time.h>

///////////////////////////////////////////////
int main(int argc, char **argv) 
{
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
#ifdef _DEBUG
	double c1, c2;
	double t_relaxation= 0.0, t_redistribute= 0.0,
	       t_propagate= 0.0, t_bounceback= 0.0;
#endif

	//Begin of the main loop
        gettimeofday( &t1, 0 );
	for (time = 0; time < properties->t_max; time++) {
		/*
		if (!(time%(properties->t_max/1))) {
			check_density(lattice, time);
		}
		*/
		
#ifdef _DEBUG
		c1= clock();
#endif
		redistribute( lattice, properties->accel, properties->density );
#ifdef _DEBUG
		c2= clock();
		t_redistribute += (c2 - c1);
#endif

#ifdef _DEBUG
		c1= clock();
#endif
		propagate(lattice);
#ifdef _DEBUG
		c2= clock();
		t_propagate += (c2 - c1);
#endif

#ifdef _DEBUG
		c1= clock();
#endif
		bounceback(lattice);
#ifdef _DEBUG
		c2= clock();
		t_bounceback += (c2 - c1);
#endif

#ifdef _DEBUG
		c1= clock();
#endif
		relaxation(lattice, properties->density, properties->omega);
#ifdef _DEBUG
		c2= clock();
		t_relaxation += (c2 - c1);
#endif
		/*
		vel = calc_velocity(lattice, time);
		printf("%d %f\n", time, vel);
		*/
	}
        gettimeofday( &t2, 0 );
        tdelta = (t2.tv_sec-t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
#ifdef _DEBUG
	fprintf( stdout, "redistribute %.6f\n", t_redistribute/CLOCKS_PER_SEC );
	fprintf( stdout, "propagate %.6f\n", t_propagate/CLOCKS_PER_SEC );
	fprintf( stdout, "bounceback %.6f\n", t_bounceback/CLOCKS_PER_SEC );
	fprintf( stdout, "relaxation %.6f\n", t_relaxation/CLOCKS_PER_SEC );
	fprintf( stdout, "total %.6f\n", t_redistribute/CLOCKS_PER_SEC +
		t_propagate/CLOCKS_PER_SEC + t_bounceback/CLOCKS_PER_SEC +
		t_relaxation/CLOCKS_PER_SEC );
#endif
      
	//execution_time = crono() - execution_time;
	comp_rey(lattice, properties, time, execution_time);
	write_results(argv[3], lattice, properties->density);
	fprintf( stdout, "%.4f\n", tdelta );
	return 0;
}
