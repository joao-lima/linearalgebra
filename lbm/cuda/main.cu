
#include <sys/time.h>
#include <stdio.h>

#include "lb.h"

int main( int argc, char **argv )
{
	struct lattice lb;
	float tdelta;
	//float vel;
	struct timeval t1, t2;
	int i;

	lb_config( &lb, argv[1], argv[2] );
	gettimeofday( &t1, 0 );
	lb_init( &lb );
	for( i= 0; i < lb.max_iter; i++ ) 
	{
		lb_redistribute( &lb );
		lb_propagate( &lb );
#if 1
		lb_bounceback( &lb );
		lb_relaxation( &lb );
#endif
		//vel = lbm.velocity( i );
		//printf( "%d %f\n", i, vel );
	}
	lb_finalize( &lb );
	lb_write_results( &lb, argv[3] );
	lb_free( &lb );
	gettimeofday( &t2, 0 );
	tdelta = (t2.tv_sec-t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
	fprintf( stdout, "%.4f\n", tdelta );

	return EXIT_SUCCESS;
}
