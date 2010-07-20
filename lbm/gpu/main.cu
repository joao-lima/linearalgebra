
#include <iostream>
#include <sys/time.h>

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
		lbm.redistribute();
		lbm.propagate();
		lbm.bounceback();
		lbm.relaxation();
		vel = lbm.velocity( i );

	}
	gettimeofday( &t2, 0 );
	tdelta = (t2.tv_sec-t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
	std::cout << "time(s): " << tdelta << std::endl;

	return 0;
}
