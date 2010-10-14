
/* Lattice Boltzmann method -- based on Schepke and Maillard 2009
   The densities are:
   	6   2  5
	  \ | /
	3 - 0 - 1
	  / | \
	7   4   8
*/


#ifndef _LB_H_
#define _LB_H_

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

struct lattice {
	int max_iter; // maximum number of iterations
	float density;
	float accel;
	float omega;
	float r_rey;

	//lattice structures
	int nx, ny, ndim, nobst;

	unsigned short *h_obst, *d_obst;
	float *h_f[9], *d_f[9];
	float *d_tf[9];

	cudaStream_t stream;
};

// Read parameters and obstacles to the structure
void lb_config( struct lattice *lb, const char *path_parameters,
		const char * path_obstacles );

// Init lattice structure and CUDA device
void lb_init( struct lattice *lb );

float lb_velocity( struct lattice *lbm, int time );

void lb_redistribute( struct lattice *lb );

void lb_propagate( struct lattice *lb );

void lb_bounceback( struct lattice *lb );

void lb_relaxation( struct lattice *lb );

void lb_sync( struct lattice *lb );

void lb_finalize( struct lattice *lb );

void lb_write_results( struct lattice *lb, const char *output );

void lb_free( struct lattice *lb );

#endif /* _LB_H_ */
