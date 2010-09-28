
#include <stdio.h>
#include <memset.h>

#include "lb.h"

static void lb_allocate( struct lattice *lb );

__device__ __host__ inline
unsigned int pos( const int x, const int y,
		const int n ) 
{
	return ( x * n + y );
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
		lb->h_obst[ pos(i,j,lb->nx) ] = 1;
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
	lb->h_data= (struct lb_d2q9*) malloc( memsize );
	memset( lb->h_data, 0, memsize );
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
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x, (lb->ny+BLOCK_SIZE-1)/threads.y );
	lb_bounceback_kernel<<< grid, threads >>>( lb->d_data, lb->d_tmp,
			lb->d_obst, lb->nx, lb->ny );
}

void lb_relaxation( struct lattice *lb )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x, (lb->ny+BLOCK_SIZE-1)/threads.y );
	lb_relaxation_kernel<<< grid, threads >>>(
			lb->d_data, lb->d_tmp, lb->d_obst,
			lb->nx, lb->ny, lb->omega );
}

void lb_finalize( struct lattice *lb )
{
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL( cudaMemcpy( lb->h_data, lb->d_data,
		lb->nx * lb->ny * sizeof(struct lb_d2q9),
		cudaMemcpyDeviceToHost));
}

void lb_write_results( const struct lattice *lb, const char *output )
{
	//local variables
	int x, y;
	int obsval;
	float u_x, u_y, d_loc, press;

	//Square speed of sound
	float c_squ = 1.0 / 3.0;

	//Open results output file
	FILE *archive = fopen( output , "w");

	//write results
	fprintf( archive, "VARIABLES = X, Y, VX, VY, PRESS, OBST\n" );
	fprintf( archive,"ZONE I= %d, J= %d, F=POINT\n", lb->nx, lb->ny );

	for( y = 0; y < lb->ny; y++ ){
		for( x = 0; x < lb->nx; x++ ){
			//if obstacle node, nothing is to do
			if ( lb->obst[pos(x,y)] == 1 ) {
				//obstacle indicator
				obsval = 1;
				//velocity components = 0
				u_x = 0.0;
				u_y = 0.0;
				//pressure = average pressure
				press = density * c_squ;
			} else {
				//integral local density
				//initialize variable d_loc
				d_loc= 0.0;
				for( i= 0; i < lb->ndim; i++ )
					d_loc += lb[ pos(pos(x,y,nx)) ][i];

				// TODO: attention: bizarre!
				// x-, and y- velocity components
				u_x = (lb->h_data[pos(x,y,lb->nx)][1] + lb->h_data[pos(x,y,lb->nx)][5] + lb->h_data[pos(x,y,lb->nx)][8] - (lb->h_data[pos(x,y,lb->nx)][3] + lb->h_data[pos(x,y,lb->nx)][6] + lb->h_data[pos(x,y,lb->nx)][7])) / d_loc;
				u_y = (lb->h_data[pos(x,y,lb->nx)][2] + lb->h_data[pos(x,y,lb->nx)][5] + lb->h_data[pos(x,y,lb->nx)][6] - (lb->h_data[pos(x,y,lb->nx)][4] + lb->h_data[pos(x,y,lb->nx)][7] + lb->h_data[pos(x,y,lb->nx)][8])) / d_loc;
				
				//pressure
				press = d_loc * c_squ;
				obsval = 0;
			}
			fprintf( archive, "%d %d %f %f %f %d\n", x, y, u_x,
				       	u_y, press, obsval );
		}
	}
	
	fclose(archive);
}

