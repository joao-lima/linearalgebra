
#include <iostream>
#include <sys/time.h>
#include <cstdio>
#include <time.h>

#include "lb.h"

int main( int argc, char **argv )
{
	lb lbm;
	double tdelta;
	struct timeval t1, t2;
#ifdef _DEBUG
	double c1, c2;
	double t_relaxation= 0.0, t_redistribute= 0.0,
	       t_propagate= 0.0, t_bounceback= 0.0;
#endif

	lbm.read( argv[1], argv[2] );
	gettimeofday( &t1, 0 );
	lbm.init();
	for( int i= 0; i < lbm.max_iteractions(); i++ ) {
#ifdef _DEBUG
		c1= clock();
#endif
		lbm.redistribute();
#ifdef _DEBUG
		cudaThreadSynchronize();
		c2= clock();
		t_redistribute += (c2 - c1);
		c1= clock();
#endif
		lbm.propagate();
#ifdef _DEBUG
		cudaThreadSynchronize();
		c2= clock();
		t_propagate += (c2 - c1);
		c1= clock();
#endif
		lbm.bounceback();
#ifdef _DEBUG
		cudaThreadSynchronize();
		c2= clock();
		t_bounceback += (c2 - c1);
		c1= clock();
#endif
		lbm.relaxation();
#ifdef _DEBUG
		cudaThreadSynchronize();
		c2= clock();
		t_relaxation += (c2 - c1);
#endif
		//vel = lbm.velocity( i );
		//printf( "%d %f\n", i, vel );
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
	fprintf( stdout, "%.4f\n", tdelta );
	lbm.write_results( argv[3] );

	return 0;
}
