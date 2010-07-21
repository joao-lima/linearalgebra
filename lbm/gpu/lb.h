
#ifndef _LB_H_
#define _LB_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

/* Lattice Boltzmann method -- based on Schepke and Maillard 2009
   The densities are:
   	6   2  5
	  \ | /
	3 - 0 - 1
	  / | \
	7   4   8
*/

class lb {
public:
	lb();

	inline int max_iteractions( void ) const
	{
		return max_iter;
	}
	void read( const char *parameters, const char *obstacles );
	void init( void );
	float velocity( int time );
	void redistribute( void );
	void propagate( void );
	void bounceback( void );
	void relaxation( void );
	void write_results( const char *file );

private:
	// Lattice
	int max_iter; // maximum number of iterations
	float density;
	float accel;
	float omega;
	float r_rey;

	//lattice structures
	int nx, ny, ndim;
	
	thrust::host_vector<float> f0, f1, f2, f3, f4, f5, f6, f7, f8;
	thrust::device_vector<float> d_f0, d_f1, d_f2, d_f3, d_f4, d_f5,
	       	d_f6, d_f7, d_f8;
	// We dont need temp data in host as it remains in GPU memory
	thrust::device_vector<float> d_tf0, d_tf1, d_tf2, d_tf3, d_tf4,
	       	d_tf5, d_tf6, d_tf7, d_tf8;
	thrust::host_vector<bool> obst;
	thrust::device_vector<bool> d_obst;

	/* returns the actual position in the array */
	inline unsigned int pos( const int x, const int y ) const
       	{
		return ( x*ny + y );
	}

	void resize( const int n );
};

#endif /* _LB_H_ */
