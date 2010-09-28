
#include <stdio.h>
#include <memset.h>

#include "lb.h"

static void lb_allocate( struct lattice *lb );

__device__ __host__ inline
unsigned int pos( const int x, const int y,
		const int n ) 
{
	return ( y * n + x );
}

void lb_config( struct lattice *lb, const char *path_parameters,
		const char * path_obstacles )
{
	FILE *f_parameters, *f_obstacles;
	int max, c=0;
	int i, j;

	f_parameters= fopen( path_parameters, "r" );
	f_obstacles= fopen( path_obstacles, "r" );
	if( (f_parameters == NULL) || (f_obstacles == NULL) )
		return;

	fscanf( f_parameters, "%d", &lb->max_iter );
	fscanf( f_parameters, "%f", &lb->density );
	fscanf( f_parameters, "%f", &lb->accel );
	fscanf( f_parameters, "%f", &lb->omega );
	fscanf( f_parameters, "%f", &lb->r_rey );

	fscanf( f_obstacles, "%d", &lb->nx );
	fscanf( f_obstacles, "%d", &lb->ny );
	fscanf( f_obstacles, "%d", &lb->ndim );
	fscanf( f_obstacles, "%d", &lb->nobst );

	lb_allocate( &lb );
	while( c < lb->nobst ){
		fscanf( f_obstacles, "%d %d", &i, &j );
		// TODO: problema aqui com indices e entrada
		lb->h_obst[ pos(i,j) ] = 1;
		c++;
	}
	fclose( f_parameters );
	fclose( f_obstacles );
}

static void lb_allocate( struct lattice *lb )
{
	unsigned int memsize;
	
	// memory for the lattice
	memsize= lb->nx * lb->ny * sizeof(struct lb_d2q9);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_data, memsize ) );
	CUDA_SAFE_CALL( cudaMemset( (void**)&lb->d_data, 0, memsize ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_tmp, memsize ) );
	CUDA_SAFE_CALL( cudaMemset( (void**)&lb->d_tmp, 0, memsize ) );

	// memory for obstacles
	memsize= lb->nobst * sizeof(int);
	lb->h_obst= (int*)malloc( memsize );
	memset( lb->h_obst, 0, memsize );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_obst, memsize ) );
}

void lb_init( struct lattice *lb )
{
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
	lb_init_kernel<<< grid, threads >>>( lb->d_data, lb->nx, lb->ny );
}


/* essa função pode ter uma implementação CUDA/thrust 
   eu vi uma função chamada transform_reduce, quem sabe ...
*/
float lb_velocity( struct lattice *lbm, int time )
{
#if 0
	int x, y, n_free;
	float u_x, d_loc;

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
#endif
	return 0;
}

void lb_redistribute( struct lattice *lb );
{
	/* here a kernel call */
	// tem de chamar esse kernel com uma dimensao apenas
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( (lb->ny+BLOCK_SIZE-1)/BLOCK_SIZE, 1 );
	redistribute_kernel<<< grid, threads >>>( lb->d_data, lb->d_obst,
		lb->accel, lb->density, lb->nx, lb->ny );
}

/*
	PROPAGATE kernel
	Authors: Joao
*/

void lb_propagate( struct lattice *lb )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
	lb_propagate_kernel<<< grid, threads >>>( lb->d_data, lb->d_tmp,
		lb->nx, lb->ny );
}

void lb_bounceback( struct lattice *lb )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (nx+BLOCK_SIZE-1)/threads.x, (ny+BLOCK_SIZE-1)/threads.y );
	lb_bounceback_kernel<<< grid, threads >>>( lb->d_data, lb->d_tmp,
			lb->d_obst, lb->nx, lb->ny );
}

/*
	RELAXATION kernel
	Authors: Antonio, Catia e Joao
*/
__global__ void relaxation_kernel( 
	float *f0, float *f1, float *f2, float *f3, float *f4, float *f5,
	float *f6, float *f7, float *f8,
	float *tf0, float *tf1, float *tf2, float *tf3, float *tf4, 
	float *tf5, float *tf6, float *tf7, float *tf8, bool* obst,
	int nx, int ny, float omega )
{
	//local variables
	float c_squ = 1.0 / 3.0;
	float t_0 = 4.0 / 9.0;
	float t_1 = 1.0 / 9.0;
	float t_2 = 1.0 / 36.0;
	float u_x, u_y;
	float u_n[9], n_equ[9], u_squ, d_loc;
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
		//u_x = (l->temp[x][y][1] + l->temp[x][y][5] + l->temp[x][y][8] - (l->temp[x][y][3] + l->temp[x][y][6] + l->temp[x][y][7])) / d_loc;

		u_y = (tf2[pos(x,y,nx)] + tf5[pos(x,y,nx)] + tf6[pos(x,y,nx)] -
				(tf4[pos(x,y,nx)] + tf7[pos(x,y,nx)] +
				 tf8[pos(x,y,nx)])) / d_loc;
		//u_y = (l->temp[x][y][2] + l->temp[x][y][5] + l->temp[x][y][6] - (l->temp[x][y][4] + l->temp[x][y][7] + l->temp[x][y][8])) / d_loc;

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
		//for (i = 0; i < l->n; i++) {
		//	l->node[x][y][i] = l->temp[x][y][i] + omega * (n_equ[i] - l->temp[x][y][i]);
		//}	
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
	float u_x, u_y, d_loc, press;

	//Square speed of sound
	float c_squ = 1.0 / 3.0;

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
				//d_loc = 0.0;
				//for (i = 0; i < 9; i++) {
				//	d_loc += l->node[x][y][i];
				//}
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
				//u_x = (l->node[x][y][1] + l->node[x][y][5] + l->node[x][y][8] - (l->node[x][y][3] + l->node[x][y][6] + l->node[x][y][7])) / d_loc;
				u_y = (f2[pos(x,y)] + f5[pos(x,y)] + f6[pos(x,y)] - (f4[pos(x,y)] + f7[pos(x,y)] + f8[pos(x,y)])) / d_loc;

				//u_y = (l->node[x][y][2] + l->node[x][y][5] + l->node[x][y][6] - (l->node[x][y][4] + l->node[x][y][7] + l->node[x][y][8])) / d_loc;
				
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

