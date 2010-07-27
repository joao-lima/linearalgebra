
#include <iostream>
#include <fstream>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "lb.h"

lb::lb() {}

__device__ inline
unsigned int pos( const int x, const int y,
		const int n ) 
{
	return ( y * n + x );
}

void lb::read( const char *parameters, const char *obstacles )
{
	std::ifstream par, obs;
	int max, c=0;
	int i, j;

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
		// TODO: problema aqui com indices e entrada
		obst[pos(i,j)] = true;
		//obst[pos(i,j)] = true;
		c++;
	}
	par.close();
	obs.close();
}

void lb::resize( const int n )
{
	f0.resize( n ); f1.resize( n ); f2.resize( n ); f3.resize( n );
       	f4.resize( n ); f5.resize( n ); f6.resize( n ); f7.resize( n );
       	f8.resize( n );
	d_tf0.resize( n ); d_tf1.resize( n ); d_tf2.resize( n ); d_tf3.resize( n );
       	d_tf4.resize( n ); d_tf5.resize( n ); d_tf6.resize( n ); d_tf7.resize( n );
       	d_tf8.resize( n );
	obst.resize( n );
	//d_obst.resize( n );
}

void lb::init( )
{
	int x, y;
	double t_0 = density * 4.0 / 9.0;
	double t_1 = density / 9.0;
	double t_2 = density / 36.0;

	//loop over computational domain
	for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
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
	// Copy of host to device
	d_f0= f0; d_f1= f1; d_f2= f2; d_f3= f3; d_f4= f4; d_f5= f5; d_f6= f6;
	d_f7= f7; d_f8= f8;
	d_obst= obst;
}

/* essa função pode ter uma implementação CUDA/thrust 
   eu vi uma função chamada transform_reduce, quem sabe ...
*/
double lb::velocity( int time ) 
{
	int x, y, n_free;
	double u_x, d_loc;

	x = nx/2;
	n_free = 0;
	u_x = 0;

	for( y = 0; y < ny; y++ ) {
		if ( obst[pos(x,y)] == false ){
			d_loc = f0[pos(x,y)];
			d_loc += f1[pos(x,y)];
			d_loc += f2[pos(x,y)];
			d_loc += f3[pos(x,y)];
			d_loc += f4[pos(x,y)];
			d_loc += f5[pos(x,y)];
			d_loc += f6[pos(x,y)];
			d_loc += f7[pos(x,y)];
			d_loc += f8[pos(x,y)];
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

/*
	REDISTRIBUTE kernel
	Authors: Catia
*/
__global__ void redistribute_kernel( double * f1, double * f3, double * f5, 
	double * f6,double * f7,double * f8, bool* obst, double accel,
       	double density, int nx, int ny ) {
    double t_1 = density * accel / 9.0;
    double t_2 = density * accel / 36.0;

    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row >= ny) return;
    if ( (obst[row * nx] == false) && ((f3[row * nx] - t_1) > 0) && 
                 ((f6[row * nx] - t_2) > 0) && (f7[row * nx] - t_2 > 0)) {
      //increase east
      f1[row * nx] += t_1;
      //decrease west
      f3[row * nx] -= t_1;
      //increase north-east
      f5[row * nx] += t_2;
      //decrease north-west
      f6[row * nx] -= t_2;
      //decrease south-west
      f7[row * nx] -= t_2;
      //increase south-east
      f8[row * nx] += t_2;
    }
}

void lb::redistribute( void )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( (ny+BLOCK_SIZE-1)/BLOCK_SIZE, 1 );
	redistribute_kernel<<< grid, threads >>>(
		thrust::raw_pointer_cast(&d_f1[0]),
		thrust::raw_pointer_cast(&d_f3[0]),
		thrust::raw_pointer_cast(&d_f5[0]),
		thrust::raw_pointer_cast(&d_f6[0]),
		thrust::raw_pointer_cast(&d_f7[0]),
		thrust::raw_pointer_cast(&d_f8[0]),
		thrust::raw_pointer_cast(&d_obst[0]),
		accel, density, nx, ny );
}

/*
	PROPAGATE kernel
	Authors: Joao
*/
__global__ void propagate_kernel( 
	double *f0, double *f1, double *f2, double *f3, double *f4, double *f5,
	double *f6, double *f7, double *f8,
	double *tf0, double *tf1, double *tf2, double *tf3, double *tf4, 
	double *tf5, double *tf6, double *tf7, double *tf8,
	int nx, int ny )
{
        //local variables
	int x_e = 0, x_w = 0, y_n = 0, y_s = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;
	//compute upper and right next neighbour nodes
	x_e = (x + 1)%nx;
	y_n = (y + 1)%ny;
	
	//compute lower and left next neighbour nodes
	x_w = (x - 1 + nx)%nx;
	y_s = (y - 1 + ny)%ny;
	
	//density propagation
	//zero
	tf0[pos(x,y,nx)] = f0[pos(x,y,nx)];
	//east
	tf1[pos(x_e,y,nx)] = f1[pos(x,y,nx)];
	//north
	tf2[pos(x,y_n,nx)] = f2[pos(x,y,nx)];
	//west
	tf3[pos(x_w,y,nx)] = f3[pos(x,y,nx)];
	//south
	tf4[pos(x,y_s,nx)] = f4[pos(x,y,nx)];
	//north-east
	tf5[pos(x_e,y_n,nx)] = f5[pos(x,y,nx)];
	//north-west
	tf6[pos(x_w,y_n,nx)] = f6[pos(x,y,nx)];
	//south-west
	tf7[pos(x_w,y_s,nx)] = f7[pos(x,y,nx)];
	//south-east
	tf8[pos(x_e,y_s,nx)] = f8[pos(x,y,nx)];
}

void lb::propagate( void )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (nx+BLOCK_SIZE-1)/threads.x, (ny+BLOCK_SIZE-1)/threads.y );
	propagate_kernel<<< grid, threads >>>(
		thrust::raw_pointer_cast(&d_f0[0]),
		thrust::raw_pointer_cast(&d_f1[0]),
		thrust::raw_pointer_cast(&d_f2[0]),
		thrust::raw_pointer_cast(&d_f3[0]),
		thrust::raw_pointer_cast(&d_f4[0]),
		thrust::raw_pointer_cast(&d_f5[0]),
		thrust::raw_pointer_cast(&d_f6[0]),
		thrust::raw_pointer_cast(&d_f7[0]),
		thrust::raw_pointer_cast(&d_f8[0]),
		// temps from here
		thrust::raw_pointer_cast(&d_tf0[0]),
		thrust::raw_pointer_cast(&d_tf1[0]),
		thrust::raw_pointer_cast(&d_tf2[0]),
		thrust::raw_pointer_cast(&d_tf3[0]),
		thrust::raw_pointer_cast(&d_tf4[0]),
		thrust::raw_pointer_cast(&d_tf5[0]),
		thrust::raw_pointer_cast(&d_tf6[0]),
		thrust::raw_pointer_cast(&d_tf7[0]),
		thrust::raw_pointer_cast(&d_tf8[0]),
		// others
		nx, ny );
}

/*
	BOUNCEBACK kernel
	Authors: Antonio
*/
__global__ void bounceback_kernel( double * f1, double * f2, double * f3,
		double * f4, double * f5, double * f6, double * f7, double * f8,
		double * tf1, double * tf2, double * tf3, double * tf4, 
		double * tf5, double * tf6, double * tf7, double * tf8,
		bool* obst, int nx, int ny) {
  //local variables
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

      if ( (row >= ny) || (col >= nx) ) return;
      if ( obst[row * nx + col] ){
		//east
		f1[row * nx + col] = tf3[row * nx + col];
		//north
		f2[row * nx + col] = tf4[row * nx + col];
		//west
		f3[row * nx + col] = tf1[row * nx + col];
		//south
		f4[row * nx + col] = tf2[row * nx + col];
		//north-east
		f5[row * nx + col] = tf7[row * nx + col];
		//north-west
		f6[row * nx + col] = tf8[row * nx + col];
		//south-west
		f7[row * nx + col] = tf5[row * nx + col];
		//south-east
		f8[row * nx + col] = tf6[row * nx + col];
      }
}

void lb::bounceback( void )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (nx+BLOCK_SIZE-1)/threads.x, (ny+BLOCK_SIZE-1)/threads.y );
	bounceback_kernel<<< grid, threads >>>(
		thrust::raw_pointer_cast(&d_f1[0]),
		thrust::raw_pointer_cast(&d_f2[0]),
		thrust::raw_pointer_cast(&d_f3[0]),
		thrust::raw_pointer_cast(&d_f4[0]),
		thrust::raw_pointer_cast(&d_f5[0]),
		thrust::raw_pointer_cast(&d_f6[0]),
		thrust::raw_pointer_cast(&d_f7[0]),
		thrust::raw_pointer_cast(&d_f8[0]),
		// temps from here
		thrust::raw_pointer_cast(&d_tf1[0]),
		thrust::raw_pointer_cast(&d_tf2[0]),
		thrust::raw_pointer_cast(&d_tf3[0]),
		thrust::raw_pointer_cast(&d_tf4[0]),
		thrust::raw_pointer_cast(&d_tf5[0]),
		thrust::raw_pointer_cast(&d_tf6[0]),
		thrust::raw_pointer_cast(&d_tf7[0]),
		thrust::raw_pointer_cast(&d_tf8[0]),
		// others
		thrust::raw_pointer_cast(&d_obst[0]),
		nx, ny );
}

/*
	RELAXATION kernel
	Authors: Antonio, Catia e Joao
*/
__global__ void relaxation_kernel( 
	double *f0, double *f1, double *f2, double *f3, double *f4, double *f5,
	double *f6, double *f7, double *f8,
	double *tf0, double *tf1, double *tf2, double *tf3, double *tf4, 
	double *tf5, double *tf6, double *tf7, double *tf8, bool* obst,
	int nx, int ny, double omega )
{
	//local variables
	double c_squ = 1.0 / 3.0;
	double t_0 = 4.0 / 9.0;
	double t_1 = 1.0 / 9.0;
	double t_2 = 1.0 / 36.0;
	double u_x, u_y;
	double u_n[9], n_equ[9], u_squ, d_loc;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;
	if ( obst[pos(x,y,nx)] == false ) {
		d_loc = tf0[pos(x,y,nx)];
		d_loc += tf1[pos(x,y,nx)];
		d_loc += tf2[pos(x,y,nx)];
		d_loc += tf3[pos(x,y,nx)];
		d_loc += tf4[pos(x,y,nx)];
		d_loc += tf5[pos(x,y,nx)];
		d_loc += tf6[pos(x,y,nx)];
		d_loc += tf7[pos(x,y,nx)];
		d_loc += tf8[pos(x,y,nx)];

		//x-, and y- velocity components
		u_x = (tf1[pos(x,y,nx)] + tf5[pos(x,y,nx)] + tf8[pos(x,y,nx)] -
				(tf3[pos(x,y,nx)] + tf6[pos(x,y,nx)] +
				 tf7[pos(x,y,nx)])) / d_loc;
		u_y = (tf2[pos(x,y,nx)] + tf5[pos(x,y,nx)] + tf6[pos(x,y,nx)] -
				(tf4[pos(x,y,nx)] + tf7[pos(x,y,nx)] +
				 tf8[pos(x,y,nx)])) / d_loc;

		//square velocity
		u_squ = u_x * u_x + u_y * u_y;

		//n- velocity compnents
		//only 3 speeds would be necessary
		u_n[1] = u_x;
		u_n[2] = u_y;
		u_n[3] = -u_x;
		u_n[4] = -u_y;
		u_n[5] = u_x + u_y;
		u_n[6] = -u_x + u_y;
		u_n[7] = -u_x - u_y;
		u_n[8] = u_x - u_y;
		
		//zero velocity density
		n_equ[0] = t_0 * d_loc * (1.0 - u_squ / (2.0 * c_squ));
		//axis speeds: factor: t_1
		n_equ[1] = t_1 * d_loc * (1.0 + u_n[1] / c_squ + u_n[1] * u_n[1] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[2] = t_1 * d_loc * (1.0 + u_n[2] / c_squ + u_n[2] * u_n[2] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[3] = t_1 * d_loc * (1.0 + u_n[3] / c_squ + u_n[3] * u_n[3] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[4] = t_1 * d_loc * (1.0 + u_n[4] / c_squ + u_n[4] * u_n[4] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));

		//diagonal speeds: factor t_2
		n_equ[5] = t_2 * d_loc * (1.0 + u_n[5] / c_squ + u_n[5] * u_n[5] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[6] = t_2 * d_loc * (1.0 + u_n[6] / c_squ + u_n[6] * u_n[6] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[7] = t_2 * d_loc * (1.0 + u_n[7] / c_squ + u_n[7] * u_n[7] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
		n_equ[8] = t_2 * d_loc * (1.0 + u_n[8] / c_squ + u_n[8] * u_n[8] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));

		
		//relaxation step
		f0[pos(x,y,nx)] = tf0[pos(x,y,nx)] + omega * (n_equ[0] - tf0[pos(x,y,nx)]);
		f1[pos(x,y,nx)] = tf1[pos(x,y,nx)] + omega * (n_equ[1] - tf1[pos(x,y,nx)]);
		f2[pos(x,y,nx)] = tf2[pos(x,y,nx)] + omega * (n_equ[2] - tf2[pos(x,y,nx)]);
		f3[pos(x,y,nx)] = tf3[pos(x,y,nx)] + omega * (n_equ[3] - tf3[pos(x,y,nx)]);
		f4[pos(x,y,nx)] = tf4[pos(x,y,nx)] + omega * (n_equ[4] - tf4[pos(x,y,nx)]);
		f5[pos(x,y,nx)] = tf5[pos(x,y,nx)] + omega * (n_equ[5] - tf5[pos(x,y,nx)]);
		f6[pos(x,y,nx)] = tf6[pos(x,y,nx)] + omega * (n_equ[6] - tf6[pos(x,y,nx)]);
		f7[pos(x,y,nx)] = tf7[pos(x,y,nx)] + omega * (n_equ[7] - tf7[pos(x,y,nx)]);
		f8[pos(x,y,nx)] = tf8[pos(x,y,nx)] + omega * (n_equ[8] - tf8[pos(x,y,nx)]);
	}
}

void lb::relaxation( void )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (nx+BLOCK_SIZE-1)/threads.x, (ny+BLOCK_SIZE-1)/threads.y );
	relaxation_kernel<<< grid, threads >>>(
		thrust::raw_pointer_cast(&d_f0[0]),
		thrust::raw_pointer_cast(&d_f1[0]),
		thrust::raw_pointer_cast(&d_f2[0]),
		thrust::raw_pointer_cast(&d_f3[0]),
		thrust::raw_pointer_cast(&d_f4[0]),
		thrust::raw_pointer_cast(&d_f5[0]),
		thrust::raw_pointer_cast(&d_f6[0]),
		thrust::raw_pointer_cast(&d_f7[0]),
		thrust::raw_pointer_cast(&d_f8[0]),
		// temps from here
		thrust::raw_pointer_cast(&d_tf0[0]),
		thrust::raw_pointer_cast(&d_tf1[0]),
		thrust::raw_pointer_cast(&d_tf2[0]),
		thrust::raw_pointer_cast(&d_tf3[0]),
		thrust::raw_pointer_cast(&d_tf4[0]),
		thrust::raw_pointer_cast(&d_tf5[0]),
		thrust::raw_pointer_cast(&d_tf6[0]),
		thrust::raw_pointer_cast(&d_tf7[0]),
		thrust::raw_pointer_cast(&d_tf8[0]),
		// others
		thrust::raw_pointer_cast(&d_obst[0]),
		nx, ny, omega );
}

void lb::write_results( const char *file ) 
{
	//local variables
	int x, y;
	bool obsval;
	double u_x, u_y, d_loc, press;

	//Square speed of sound
	double c_squ = 1.0 / 3.0;

	//Open results output file
	FILE *archive = fopen(file, "w");

	//write results
	fprintf( archive, "VARIABLES = X, Y, VX, VY, PRESS, OBST\n" );
	fprintf( archive,"ZONE I= %d, J= %d, F=POINT\n", nx, ny );

	thrust::copy( d_f0.begin(), d_f0.end(), f0.begin() );
	thrust::copy( d_f1.begin(), d_f1.end(), f1.begin() );
	thrust::copy( d_f2.begin(), d_f2.end(), f2.begin() );
	thrust::copy( d_f3.begin(), d_f3.end(), f3.begin() );
	thrust::copy( d_f4.begin(), d_f4.end(), f4.begin() );
	thrust::copy( d_f5.begin(), d_f5.end(), f5.begin() );
	thrust::copy( d_f6.begin(), d_f6.end(), f6.begin() );
	thrust::copy( d_f7.begin(), d_f7.end(), f7.begin() );
	thrust::copy( d_f8.begin(), d_f8.end(), f8.begin() );
	for( y = 0; y < ny; y++ ){
		for( x = 0; x < nx; x++ ){
			//if obstacle node, nothing is to do
			if (obst[pos(x,y)] == true) {
				//obstacle indicator
				obsval = true;
				//velocity components = 0
				u_x = 0.0;
				u_y = 0.0;
				//pressure = average pressure
				press = density * c_squ;
			} else {
				//integral local density
				//initialize variable d_loc
				d_loc = f0[pos(x,y)];
				d_loc += f1[pos(x,y)];
				d_loc += f2[pos(x,y)];
				d_loc += f3[pos(x,y)];
				d_loc += f4[pos(x,y)];
				d_loc += f5[pos(x,y)];
				d_loc += f6[pos(x,y)];
				d_loc += f7[pos(x,y)];
				d_loc += f8[pos(x,y)];
				// x-, and y- velocity components
				u_x = (f1[pos(x,y)] + f5[pos(x,y)] + f8[pos(x,y)] - (f3[pos(x,y)] + f6[pos(x,y)] + f7[pos(x,y)])) / d_loc;
				u_y = (f2[pos(x,y)] + f5[pos(x,y)] + f6[pos(x,y)] - (f4[pos(x,y)] + f7[pos(x,y)] + f8[pos(x,y)])) / d_loc;

				
				//pressure
				press = d_loc * c_squ;
				obsval = false;
			}
			fprintf( archive, "%d %d %f %f %f %d\n", x, y, u_x,
				       	u_y, press, obsval );
		}
	}
	
	fclose(archive);
}

