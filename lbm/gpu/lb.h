
#ifndef _LB_H_
#define _LB_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Lattice Boltzmann method -- based on Schepke and Maillard 2009
   The densities are:
   	6   2  5
	  \ | /
	3 - 0 - 1
	  / | \
	7   4   8
*/

class lb {
	lb();
	void read( const char *parameters, const char *obstacles );
	void init( void );
private:
	// Lattice
	int max_iter; // maximum number of iterations
	float density;
	float accel;
	float omega;
	float r_rey;

	//lattice structres
	int nx, ny, ndim;
	
	thrust::host_vector<float> f0, f1, f2, f3, f4, f5, f6, f7, f8;
	thrust::host_vector<bool> obst;

	inline unsigned int pos( const int x, const int y ) const
       	{
		return ( x*nx + y );
	}

	void resize( const int n );
};

#endif /* _LB_H_ */
