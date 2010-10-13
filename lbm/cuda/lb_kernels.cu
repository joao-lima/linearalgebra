
__device__ inline
unsigned int pos( const int x, const int y,
		const int N ) 
{
	return ( y * N + x );
}

__global__ void lb_init_kernel( 
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
		const int nx, const int ny, const float density )
{
	float t_0 = density * 4.0 / 9.0;
	float t_1 = density / 9.0;
	float t_2 = density / 36.0;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;

	//zero velocity density
	f0[ pos(x,y,nx) ] = t_0;
	//equilibrium densities for axis speeds
	f1[ pos(x,y,nx) ] = t_1;
	f2[ pos(x,y,nx) ] = t_1;
	f3[ pos(x,y,nx) ] = t_1;
	f4[ pos(x,y,nx) ] = t_1;
	//equilibrium densities for diagonal speeds
	f5[ pos(x,y,nx) ] = t_2;
	f6[ pos(x,y,nx) ] = t_2;
	f7[ pos(x,y,nx) ] = t_2;
	f8[ pos(x,y,nx) ] = t_2;
}

__global__ void lb_redistribute_kernel( 
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	const unsigned short *obst, const float accel, const float density,
	const int nx, const int ny ) 
{
	float t_1 = density * accel / 9.0;
	float t_2 = density * accel / 36.0;

	int row = blockIdx.x * blockDim.x + threadIdx.x; 
	if (row >= ny) return;
	if ( (obst[row * nx] == 0) && ((f3[row * nx] - t_1) > 0) && 
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

__global__ void lb_propagate_kernel( 
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	float *tf0, float *tf1, float *tf2, float *tf3, float *tf4,
	float *tf5, float *tf6, float *tf7, float *tf8,
	const int nx, const int ny)
{
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

__global__ void lb_bounceback_kernel(
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	const float *tf0, const float *tf1, const float *tf2, const float *tf3,
	const float *tf4, const float *tf5, const float *tf6, const float *tf7,
	const float *tf8,
	const unsigned short *obst,
	const int nx, const int ny )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x; 

	if ( (row >= ny) || (col >= nx) ) return;

	if ( obst[row * nx + col] == 1 ){
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

__global__ void lb_relaxation_kernel( 
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	const float *tf0, const float *tf1, const float *tf2,
	const float *tf3, const float *tf4,
	const float *tf5, const float *tf6, const float *tf7, const float *tf8,
		const unsigned short *obst, const int nx, const int ny,
		const float omega )
{
	float c_squ = 1.0 / 3.0;
	float t_0 = 4.0 / 9.0;
	float t_1 = 1.0 / 9.0;
	float t_2 = 1.0 / 36.0;
	float u_x, u_y;
	float u_n[9], n_equ[9], u_squ, d_loc;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;
	if ( obst[pos(x,y,nx)] == 0 ) {
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
