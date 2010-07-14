
#include <iostream>
#include <fstream>

#include "lb.h"

lb::lb() {}

void lb::read( const char *parameters, const char *obstacles )
{
	std::ifstream par, obs;
	int max, c=0;
	par.open( parameters );
	obs.open( obstacles );
	if( !par.is_open() || !obs.is_open() )
		return;
	par >> max_iter;
	par >> density;
	par >> accel;
	par >> omega;
	par >> r_rey;

	obs >> nx;
	obs >> ny;
	obs >> ndim;
	obs >> max;

	resize( nx * ny );
	while( c < max ){
		obs >> i;
		obs >> j;
		obst[pos(i-1,j-1)] = true;
		c++;
	}
	par.close();
	obs.close();
}

void lb::resize( const int n )
{
	f0.resize( n ); obst.resize( n );
	f1.resize( n ); f2.resize( n ); f3.resize( n ); f4.resize( n );
	f5.resize( n ); f6.resize( n ); f7.resize( n ); f8.resize( n );
}

void lb::init( )
{
	int x, y;
	double t_0 = density * 4.0 / 9.0;
	double t_1 = density / 9.0;
	double t_2 = density / 36.0;

	//loop over computational domain
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			//zero velocity density
			f0[pos(x,y)] = t_0;
			//equilibrium densities for axis speeds
			f1[pos(x,y)] = t_1;
			f2[pos(x,y)] = t_1;
			f3[pos(x,y)] = t_1;
			f4[pos(x,y)] = t_1;
			//equilibrium densities for diagonal speeds
			f5[pos(x,y)] = t_2;
			f6[pos(x,y)] = t_2;
			f7[pos(x,y)] = t_2;
			f8[pos(x,y)] = t_2;
		}
	}
}

float lb::velocity( int time ) 
{
	int x, y, i, n_free;
	double u_x, d_loc;

	x = nx/2;
	n_free = 0;
	u_x = 0;

	for( y = 0; y < ny; y++ ) {
		if ( obst[pos(x,y)] == false ){
			d_loc = d_loc + f0[pos(x,y)];
			d_loc += d_loc + f1[pos(x,y)];
			d_loc += d_loc + f2[pos(x,y)];
			d_loc += d_loc + f3[pos(x,y)];
			d_loc += d_loc + f4[pos(x,y)];
			d_loc += d_loc + f5[pos(x,y)];
			d_loc += d_loc + f6[pos(x,y)];
			d_loc += d_loc + f7[pos(x,y)];
			d_loc += d_loc + f8[pos(x,y)];
			u_x = u_x + (f1[pos(x,y)]
				 + f5[pos(x,y)] + f8[pos(x,y)] - 
				 (f3[pos(x,y)] + f6[pos(x,y)]
				  + f7[pos(x,y)])) / d_loc;
			n_free++;
		}
	}
	/*
	//Optional
	if (time%500 == 0) {
		FILE *c = fopen("convergence9.out", "a");
		fprintf(c, "%d %lf\n", time, u_x / n_free);
		fclose(c);
	}
	*/
	return u_x / n_free;
}
