
__device__ inline
unsigned int pos( const int x, const int y,
		const int n ) 
{
	return ( x * n + y );
}

__global__ void lb_init_kernel( struct lb_d2q9 *lb, const int nx,
		const int ny, const float density )
{
	float t_0 = density * 4.0 / 9.0;
	float t_1 = density / 9.0;
	float t_2 = density / 36.0;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;

	//zero velocity density
	lb[ pos(x,y,nx) ].d[0] = t_0;
	//equilibrium densities for axis speeds
	lb[ pos(x,y,nx) ].d[1] = t_1;
	lb[ pos(x,y,nx) ].d[2] = t_1;
	lb[ pos(x,y,nx) ].d[3] = t_1;
	lb[ pos(x,y,nx) ].d[4] = t_1;
	//equilibrium densities for diagonal speeds
	lb[ pos(x,y,nx) ].d[5] = t_2;
	lb[ pos(x,y,nx) ].d[6] = t_2;
	lb[ pos(x,y,nx) ].d[7] = t_2;
	lb[ pos(x,y,nx) ].d[8] = t_2;
}

__global__ void lb_redistribute_kernel( struct lb_d2q9 *lb,
	const unsigned short *obst, const float accel, const float density,
	const int nx, const int ny ) 
{
    //nx e ny sao as dimensoes
    //local variables
    float t_1 = density * accel / 9.0;
    float t_2 = density * accel / 36.0;
    int x = blockIdx.x * blockDim.x + threadIdx.x; 

    if (x >= ny) return;
    //check to avoid negative densities
    //check false | true
    if ( (obst[x * nx] == 0) && ((lb[x * nx].d[3] - t_1) > 0) && 
                 ((lb[x * nx].d[6] - t_2) > 0) && 
		 (lb[x * nx].d[7] - t_2 > 0)) {
      //increase east
      lb[x * nx].d[1] += t_1;
      //decrease west
      lb[x * nx].d[3] -= t_1;
      //increase north-east
      lb[x * nx].d[5] += t_2;
      //decrease north-west
      lb[x * nx].d[6] -= t_2;
      //decrease south-west
      lb[x * nx].d[7] -= t_2;
      //increase south-east
      lb[x * nx].d[8] += t_2;
    }
}

__global__ void lb_propagate_kernel( struct lb_d2q9 *lb,
		struct lb_d2q9 *tmp, const int nx, const int ny)
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
	tmp[ pos(x,y,nx) ].d[0] = lb[ pos(x,y,nx) ].d[0];
	//east
	tmp[ pos(x_e,y,nx) ].d[1] = lb[ pos(x,y,nx) ].d[1];
	//north
	tmp[ pos(x,y_n,nx) ].d[2] = lb[ pos(x,y,nx) ].d[2];
	//west
	tmp[ pos(x_w,y,nx) ].d[3] = lb[ pos(x,y,nx) ].d[3];
	//south
	tmp[ pos(x,y_s,nx) ].d[4] = lb[ pos(x,y,nx) ].d[4];
	//north-east
	tmp[ pos(x_e,y_n,nx) ].d[5] = lb[ pos(x,y,nx) ].d[5];
	//north-west
	tmp[ pos(x_w,y_n,nx) ].d[6] = lb[ pos(x,y,nx) ].d[6];
	//south-west
	tmp[ pos(x_w,y_s,nx) ].d[7] = lb[ pos(x,y,nx) ].d[7];
	//south-east
	tmp[ pos(x_e,y_s,nx) ].d[8] = lb[ pos(x,y,nx) ].d[8];
}

__global__ void lb_bounceback_kernel( struct lb_d2q9 *lb,
		const struct lb_d2q9 *tmp, const unsigned short *obst,
		const int nx, const int ny )
{
	int x = blockIdx.y * blockDim.y + threadIdx.y; 
	int y = blockIdx.x * blockDim.x + threadIdx.x; 

	if ( (x >= ny) || (y >= nx) ) return;

	if ( obst[ pos(x,y,nx) ] == 1 ){
		//east
		lb[ pos(x,y,nx) ].d[1] = tmp[ pos(x,y,nx) ].d[3];
		//north
		lb[ pos(x,y,nx) ].d[2] = tmp[ pos(x,y,nx) ].d[4];
		//west
		lb[ pos(x,y,nx) ].d[3] = tmp[ pos(x,y,nx) ].d[1];
		//south
		lb[ pos(x,y,nx) ].d[4] = tmp[ pos(x,y,nx) ].d[2];
		//north-east
		lb[ pos(x,y,nx) ].d[5] = tmp[ pos(x,y,nx) ].d[7];
		//north-west
		lb[ pos(x,y,nx) ].d[6] = tmp[ pos(x,y,nx) ].d[8];
		//south-west
		lb[ pos(x,y,nx) ].d[7] = tmp[ pos(x,y,nx) ].d[5];
		//south-east
		lb[ pos(x,y,nx) ].d[8] = tmp[ pos(x,y,nx) ].d[6];
	}
}

__global__ void lb_relaxation_kernel( 
		struct lb_d2q9 *lb, const struct lb_d2q9 *tmp,
		const unsigned short *obst, const int nx, const int ny,
		const float omega )
{
	const float c_squ = 1.0 / 3.0;
	const float t_0 = 4.0 / 9.0;
	const float t_1 = 1.0 / 9.0;
	const float t_2 = 1.0 / 36.0;
	float u_x, u_y;
	float u_n[9], n_equ[9], u_squ, d_loc;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int i;

	if( (y >= ny) || (x >= nx) ) return;

	if ( obst[pos(x,y,nx)] == 0 ) {
		d_loc= 0.0;
		for (i = 0; i < 9; i++)
			d_loc += tmp[ pos(x,y,nx) ].d[i];

		//x-, and y- velocity components
		u_x = (tmp[pos(x,y,nx)].d[1] + tmp[pos(x,y,nx)].d[5] + tmp[pos(x,y,nx)].d[8] -
				(tmp[pos(x,y,nx)].d[3] + tmp[pos(x,y,nx)].d[6] +
				 tmp[pos(x,y,nx)].d[7])) / d_loc;

		u_y = (tmp[pos(x,y,nx)].d[2] + tmp[pos(x,y,nx)].d[5] + tmp[pos(x,y,nx)].d[6] -
				(tmp[pos(x,y,nx)].d[4] + tmp[pos(x,y,nx)].d[7] +
				 tmp[pos(x,y,nx)].d[8])) / d_loc;

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

		for (i = 0; i < 9; i++)
			lb[pos(x,y,nx)].d[i] = tmp[pos(x,y,nx)].d[i]
			       	+ omega * (n_equ[i] - tmp[pos(x,y,nx)].d[i]);
	}
}

