
#include <stdio.h>
#include <string.h>

#include "lb.h"
#include "cuda_safe.h"
#include "lb_kernels.cu"

#define POS(x,y,N)	((y-1)*N+(x-1))

static void lb_allocate( struct lattice *lb );

void lb_config( struct lattice *lb, const char *path_parameters,
		const char * path_obstacles )
{
	FILE *f_parameters, *f_obstacles;
	int c=0;
	int i, j;

#ifdef _DEBUG
	fprintf( stdout, "lb_config\n" );
	fflush(stdout);
#endif
	f_parameters= fopen( path_parameters, "r" );
	f_obstacles= fopen( path_obstacles, "r" );
	if( (f_parameters == NULL) || (f_obstacles == NULL) ) {
		fprintf( stderr, "No file found\n" );
		fflush( stderr );
		exit( EXIT_FAILURE );
	}

	fscanf( f_parameters, "%d", &lb->max_iter );
	fscanf( f_parameters, "%f", &lb->density );
	fscanf( f_parameters, "%f", &lb->accel );
	fscanf( f_parameters, "%f", &lb->omega );
	fscanf( f_parameters, "%f", &lb->r_rey );

	fscanf( f_obstacles, "%d", &lb->nx );
	fscanf( f_obstacles, "%d", &lb->ny );
	fscanf( f_obstacles, "%d", &lb->ndim );
	fscanf( f_obstacles, "%d", &lb->nobst );

	fprintf( stdout, "nx=%d ny=%d ndim=%d maxiter=%d nobst=%d\n",
	      lb->nx, lb->ny, lb->ndim, lb->max_iter, lb->nobst );
	fflush( stdout );
	lb_allocate( lb );
	while( c < lb->nobst ){
		fscanf( f_obstacles, "%d %d", &i, &j );
		lb->h_obst[ POS(i,j,lb->nx) ] = 1;
		c++;
	}
	fclose( f_parameters );
	fclose( f_obstacles );
}

static void lb_allocate( struct lattice *lb )
{
	unsigned int memsize;
	
#ifdef _DEBUG
	fprintf( stdout, "lb_allocate\n" );
	fflush(stdout);
#endif
	// memory for the lattice
	memsize= lb->nx * lb->ny * sizeof(struct lb_d2q9);
	lb->h_data= (struct lb_d2q9*) malloc( memsize );
	memset( lb->h_data, 0, memsize );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_data, memsize ) );
	CUDA_SAFE_CALL( cudaMemset( lb->d_data, 0, memsize ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_tmp, memsize ) );
	CUDA_SAFE_CALL( cudaMemset( lb->d_tmp, 0, memsize ) );

	// memory for obstacles
	memsize= lb->nobst * sizeof(unsigned short);
	lb->h_obst= (unsigned short*) malloc( memsize );
	memset( lb->h_obst, 0, memsize );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_obst, memsize ) );
}

void lb_init( struct lattice *lb )
{
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_init\n" );
	fflush(stdout);
#endif
	CUDA_SAFE_CALL( cudaMemcpy( lb->d_obst, lb->h_obst,
		lb->nobst * sizeof(unsigned short), cudaMemcpyHostToDevice) );
	lb_init_kernel<<< grid, threads >>>( lb->d_data, lb->nx, lb->ny,
		       lb->density );
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

void lb_redistribute( struct lattice *lb )
{
	/* here a kernel call */
	// tem de chamar esse kernel com uma dimensao apenas
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( (lb->ny+BLOCK_SIZE-1)/BLOCK_SIZE, 1 );
#ifdef _DEBUG
	fprintf( stdout, "lb_redistribute\n" );
	fflush(stdout);
#endif
	lb_redistribute_kernel<<< grid, threads >>>( lb->d_data, lb->d_obst,
		lb->accel, lb->density, lb->nx, lb->ny );
}

void lb_propagate( struct lattice *lb )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_propagate\n" );
	fflush(stdout);
#endif
	lb_propagate_kernel<<< grid, threads >>>( lb->d_data, lb->d_tmp,
		lb->nx, lb->ny );
}

void lb_bounceback( struct lattice *lb )
{
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x, (lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_bounceback\n" );
	fflush(stdout);
#endif
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
#ifdef _DEBUG
	fprintf( stdout, "lb_finalize\n" );
	fflush(stdout);
#endif
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL( cudaMemcpy( lb->h_data, lb->d_data,
		lb->nx * lb->ny * sizeof(struct lb_d2q9),
		cudaMemcpyDeviceToHost));
}

void lb_write_results( struct lattice *lb, const char *output )
{
	int x, y, i;
	int obsval;
	float u_x, u_y, d_loc, press;

	//Square speed of sound
	float c_squ = 1.0 / 3.0;

#ifdef _DEBUG
	fprintf( stdout, "lb_write_results\n" );
	fflush(stdout);
#endif
	//Open results output file
	FILE *archive = fopen( output , "w");

	//write results
	fprintf( archive, "VARIABLES = X, Y, VX, VY, PRESS, OBST\n" );
	fprintf( archive,"ZONE I= %d, J= %d, F=POINT\n", lb->nx, lb->ny );

	for( y = 0; y < lb->ny; y++ ){
		for( x = 0; x < lb->nx; x++ ){
			//if obstacle node, nothing is to do
			if ( lb->h_obst[POS(x,y,lb->nx)] == 1 ) {
				//obstacle indicator
				obsval = 1;
				//velocity components = 0
				u_x = 0.0;
				u_y = 0.0;
				//pressure = average pressure
				press = lb->density * c_squ;
			} else {
				//integral local density
				//initialize variable d_loc
				d_loc= 0.0;
				for( i= 0; i < lb->ndim; i++ )
					d_loc += lb->h_data[ POS(x,y,lb->nx) ].d[i];

				// TODO: attention: bizarre!
				// x-, and y- velocity components
				u_x = (lb->h_data[POS(x,y,lb->nx)].d[1] + lb->h_data[POS(x,y,lb->nx)].d[5] + lb->h_data[POS(x,y,lb->nx)].d[8] - (lb->h_data[POS(x,y,lb->nx)].d[3] + lb->h_data[POS(x,y,lb->nx)].d[6] + lb->h_data[POS(x,y,lb->nx)].d[7])) / d_loc;
				u_y = (lb->h_data[POS(x,y,lb->nx)].d[2] + lb->h_data[POS(x,y,lb->nx)].d[5] + lb->h_data[POS(x,y,lb->nx)].d[6] - (lb->h_data[POS(x,y,lb->nx)].d[4] + lb->h_data[POS(x,y,lb->nx)].d[7] + lb->h_data[POS(x,y,lb->nx)].d[8])) / d_loc;
				
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

void lb_free( struct lattice *lb )
{
#ifdef _DEBUG
	fprintf( stdout, "lb_free\n" );
	fflush(stdout);
#endif
	CUDA_SAFE_CALL( cudaFree( lb->d_data ) );
	CUDA_SAFE_CALL( cudaFree( lb->d_tmp ) );
	free( lb->h_data );

	free( lb->h_obst );
	CUDA_SAFE_CALL( cudaFree( lb->d_obst ) );
	CUDA_SAFE_CALL( cudaThreadExit() );
	fprintf(stdout,"bazzinga!\n"); fflush(stdout);
}
