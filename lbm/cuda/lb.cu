
#include <stdio.h>
#include <string.h>

#include "lb.h"
#include "cuda_safe.h"
#include "lb_kernels.cu"

#define OBJ(X,Y,N)	((Y-1)*N+(X-1))
#define POS(X,Y,N)	(Y*N+X)

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
		lb->h_obst[ OBJ(i,j,lb->nx) ] = 1;
		c++;
	}
	fclose( f_parameters );
	fclose( f_obstacles );
}

static void lb_allocate( struct lattice *lb )
{
	unsigned int mem_size;
	int i;
	
#ifdef _DEBUG
	fprintf( stdout, "lb_allocate\n" );
	fflush(stdout);
#endif
	// memory for the lattice
	mem_size= lb->nx * lb->ny * sizeof(float);
	for( i= 0; i < lb->ndim; i++ ) {
		lb->h_f[i]= (float*) malloc( mem_size );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_f[i], mem_size ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_tf[i], mem_size ) );
	}

	// memory for obstacles
	mem_size= lb->nx * lb->ny * sizeof(unsigned short);
	lb->h_obst= (unsigned short*) malloc( mem_size );
	memset( lb->h_obst, 0, mem_size );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&lb->d_obst, mem_size ) );
}

void lb_init( struct lattice *lb )
{
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x, (lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_init\n" );
	fflush(stdout);
#endif
	lb_init_kernel<<< grid, threads >>>( 
		lb->d_f[0], lb->d_f[1], lb->d_f[2], lb->d_f[3], lb->d_f[4], 
		lb->d_f[5], lb->d_f[6], lb->d_f[7], lb->d_f[8],
			lb->nx, lb->ny, lb->density );
	CUDA_SAFE_THREAD_SYNC();
	CUDA_SAFE_CALL( cudaMemcpy( lb->d_obst, lb->h_obst,
		lb->nx * lb->ny * sizeof(unsigned short),
	       	cudaMemcpyHostToDevice) );
#if 0
	int x, y, i;
	float t_0 = lb->density * 4.0 / 9.0;
	float t_1 = lb->density / 9.0;
	float t_2 = lb->density / 36.0;
	for( x= 0; x < lb->nx; x++) {
		for( y= 0; y < lb->ny; y++ ){
		//zero velocity density
		lb->h_f[0][ POS(x,y,lb->nx) ] = t_0;
		//equilibrium densities for axis speeds
		lb->h_f[1][ POS(x,y,lb->nx) ] = t_1;
		lb->h_f[2][ POS(x,y,lb->nx) ] = t_1;
		lb->h_f[3][ POS(x,y,lb->nx) ] = t_1;
		lb->h_f[4][ POS(x,y,lb->nx) ] = t_1;
		//equilibrium densities for diagonal speeds
		lb->h_f[5][ POS(x,y,lb->nx) ] = t_2;
		lb->h_f[6][ POS(x,y,lb->nx) ] = t_2;
		lb->h_f[7][ POS(x,y,lb->nx) ] = t_2;
		lb->h_f[8][ POS(x,y,lb->nx) ] = t_2;
		}
	}
	for( i= 0; i < lb->ndim; i++ )
		CUDA_SAFE_CALL( cudaMemcpy( lb->d_f[i], lb->h_f[i],
			lb->nx * lb->ny * sizeof(float),
			cudaMemcpyHostToDevice) );
#endif
}


float lb_velocity( struct lattice *lbm, int time )
{
	(void) lbm;
	(void) time;
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
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( (lb->ny+BLOCK_SIZE-1)/BLOCK_SIZE, 1 );
#ifdef _DEBUG
	fprintf( stdout, "lb_redistribute\n" );
	fflush(stdout);
#endif
	lb_redistribute_kernel<<< grid, threads >>>(
		lb->d_f[0], lb->d_f[1], lb->d_f[2], lb->d_f[3], lb->d_f[4], 
		lb->d_f[5], lb->d_f[6], lb->d_f[7], lb->d_f[8], lb->d_obst,
		lb->accel, lb->density, lb->nx, lb->ny );
	CUDA_SAFE_THREAD_SYNC();
}

void lb_propagate( struct lattice *lb )
{
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_propagate\n" );
	fflush(stdout);
#endif
	lb_propagate_kernel<<< grid, threads >>>( 
		lb->d_f[0], lb->d_f[1], lb->d_f[2], lb->d_f[3], lb->d_f[4], 
		lb->d_f[5], lb->d_f[6], lb->d_f[7], lb->d_f[8], 
		lb->d_tf[0], lb->d_tf[1], lb->d_tf[2], lb->d_tf[3],
		lb->d_tf[4], lb->d_tf[5], lb->d_tf[6], lb->d_tf[7],
		lb->d_tf[8], lb->nx, lb->ny );
	CUDA_SAFE_THREAD_SYNC();
}

void lb_bounceback( struct lattice *lb )
{
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x, (lb->ny+BLOCK_SIZE-1)/threads.y );
#ifdef _DEBUG
	fprintf( stdout, "lb_bounceback\n" );
	fflush(stdout);
#endif
	lb_bounceback_kernel<<< grid, threads >>>( lb->d_f[0], lb->d_f[1], 
		lb->d_f[2], lb->d_f[3], lb->d_f[4], lb->d_f[5], lb->d_f[6], 
		lb->d_f[7], lb->d_f[8],
		lb->d_tf[0], lb->d_tf[1], lb->d_tf[2], lb->d_tf[3], 
		lb->d_tf[4], lb->d_tf[5], lb->d_tf[6], lb->d_tf[7],
		lb->d_tf[8], lb->d_obst, lb->nx, lb->ny );
	CUDA_SAFE_THREAD_SYNC();
}

void lb_relaxation( struct lattice *lb )
{
#if 1
	/* here a kernel call */
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	dim3 grid( (lb->nx+BLOCK_SIZE-1)/threads.x,
			(lb->ny+BLOCK_SIZE-1)/threads.y );
	lb_relaxation_kernel<<< grid, threads >>>( lb->d_f[0], lb->d_f[1], 
		lb->d_f[2], lb->d_f[3], lb->d_f[4], lb->d_f[5], lb->d_f[6], 
		lb->d_f[7], lb->d_f[8],
		lb->d_tf[0], lb->d_tf[1], lb->d_tf[2], lb->d_tf[3], 
		lb->d_tf[4], lb->d_tf[5], lb->d_tf[6], lb->d_tf[7], 
		lb->d_tf[8], lb->d_obst,
		lb->nx, lb->ny, lb->omega );
	CUDA_SAFE_THREAD_SYNC();
#endif
}

void lb_finalize( struct lattice *lb )
{
	int i;
#ifdef _DEBUG
	fprintf( stdout, "lb_finalize\n" );
	fflush(stdout);
#endif
	for( i= 0; i < lb->ndim; i++ )
		CUDA_SAFE_CALL( cudaMemcpy( lb->h_f[i], lb->d_f[i],
			lb->nx * lb->ny * sizeof(float),
			cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
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
				d_loc= 0.0f;
				for( i= 0; i < lb->ndim; i++ )
					d_loc += lb->h_f[i][ POS(x,y,lb->nx) ];

#define NODE(X,Y,D)		(lb->h_f[D][(Y*lb->nx+X)])
				// x-, and y- velocity components
				u_x = (NODE(x,y,1) + NODE(x,y,5) + NODE(x,y,8) - (NODE(x,y,3) + NODE(x,y,6) + NODE(x,y,7))) / d_loc;
				u_y = (NODE(x,y,2) + NODE(x,y,5) + NODE(x,y,6) - (NODE(x,y,4) + NODE(x,y,7) + NODE(x,y,8))) / d_loc;
				
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
	int i;
	for( i= 0; i < lb->ndim; i++ ) {
		CUDA_SAFE_CALL( cudaFree( lb->d_f[i] ) );
		CUDA_SAFE_CALL( cudaFree( lb->d_tf[i] ) );
		free( lb->h_f[i] );
	}

	free( lb->h_obst );
	CUDA_SAFE_CALL( cudaFree( lb->d_obst ) );
	CUDA_SAFE_CALL( cudaThreadExit() );
#ifdef _DEBUG
	fprintf(stdout,"bazzinga!\n"); fflush(stdout);
#endif
}
