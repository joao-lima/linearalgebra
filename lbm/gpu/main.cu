
#include <iostream>
#include <sys/time.h>
#include <cstdio>

#include "lb.h"

int main( int argc, char **argv )
{
	lb lbm;
	double tdelta;
	struct timeval t1, t2;

	lbm.read( argv[1], argv[2] );
	gettimeofday( &t1, 0 );
	lbm.init();
	for( int i= 0; i < lbm.max_iteractions(); i++ ) {
#ifdef _DEBUG
#endif
		lbm.redistribute();
#ifdef _DEBUG
		cudaThreadSynchronize();
#endif
		lbm.propagate();
#ifdef _DEBUG
		cudaThreadSynchronize();
#endif
		lbm.bounceback();
#ifdef _DEBUG
		cudaThreadSynchronize();
#endif
		lbm.relaxation();
#ifdef _DEBUG
		cudaThreadSynchronize();
#endif
		//vel = lbm.velocity( i );
		//printf( "%d %f\n", i, vel );
	}
	gettimeofday( &t2, 0 );
	tdelta = (t2.tv_sec-t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
	fprintf( stdout, "%.4f\n", tdelta );
	lbm.write_results( argv[3] );

	return 0;
}
