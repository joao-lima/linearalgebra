
__global__ void lb_init_kernel( struct lb_d2q9 *lb, const int nx,
		const int ny )
{
	float t_0 = density * 4.0 / 9.0;
	float t_1 = density / 9.0;
	float t_2 = density / 36.0;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y >= ny) || (x >= nx) ) return;

	//zero velocity density
	lb[ pos(x,y) ][0] = t_0;
	//equilibrium densities for axis speeds
	lb[ pos(x,y) ][1] = t_1;
	lb[ pos(x,y) ][2] = t_1;
	lb[ pos(x,y) ][3] = t_1;
	lb[ pos(x,y) ][4] = t_1;
	//equilibrium densities for diagonal speeds
	lb[ pos(x,y) ][5] = t_2;
	lb[ pos(x,y) ][6] = t_2;
	lb[ pos(x,y) ][7] = t_2;
	lb[ pos(x,y) ][8] = t_2;
}

__global__ void lb_redistribute_kernel( struct lb_d2q9 *lb,
	const int *obst, const float accel, const float density,
	const int nx, const int ny ) 
{
    //nx e ny sao as dimensoes
    //local variables
    float t_1 = density * accel / 9.0;
    float t_2 = density * accel / 36.0;
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row >= ny) return;
    //check to avoid negative densities
    //check false | true
    if ( (obst[row * nx] == 0) && ((lb[row * nx][3] - t_1) > 0) && 
                 ((lb[row * nx][6] - t_2) > 0) && 
		 (lb[row * nx][7] - t_2 > 0)) {
      //increase east
      lb[row * nx][1] += t_1;
      //decrease west
      lb[row * nx][3] -= t_1;
      //increase north-east
      lb[row * nx][5] += t_2;
      //decrease north-west
      lb[row * nx][6] -= t_2;
      //decrease south-west
      lb[row * nx][7] -= t_2;
      //increase south-east
      lb[row * nx][8] += t_2;
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
	tmp[ pos(x,y,nx) ][0] = lb[ pos(x,y,nx) ][0];
	//east
	tmp[ pos(x_e,y,nx) ][1] = lb[ pos(x,y,nx) ][1];
	//north
	tmp[ pos(x,y_n,nx) ][2] = lb[ pos(x,y,nx) ][2];
	//west
	tmp[ pos(x_w,y,nx) ][3] = lb[ pos(x,y,nx) ][3];
	//south
	tmp[ pos(x,y_s,nx) ][4] = lb[ pos(x,y,nx) ][4];
	//north-east
	tmp[ pos(x_e,y_n,nx) ][5] = lb[ pos(x,y,nx) ][5];
	//north-west
	tmp[ pos(x_w,y_n,nx) ][6] = lb[ pos(x,y,nx) ][6];
	//south-west
	tmp[ pos(x_w,y_s,nx) ][7] = lb[ pos(x,y,nx) ][7];
	//south-east
	tmp[ pos(x_e,y_s,nx) ][8] = lb[ pos(x,y,nx) ][8];
}

__global__ void lb_bounceback_kernel( struct lb_d2q9 *lb,
		struct lb_d2q9 *tmp, const int *obst,
		const int nx, const int ny )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x; 

	if ( (row >= ny) || (col >= nx) ) return;

	if ( obst[row * nx + col] == 1 ){
		//east
		f1[row * nx + col][] = tf3[row * nx + col];
		//north
		f2[row * nx + col][] = tf4[row * nx + col];
		//west
		f3[row * nx + col][] = tf1[row * nx + col];
		//south
		f4[row * nx + col][] = tf2[row * nx + col];
		//north-east
		f5[row * nx + col][] = tf7[row * nx + col];
		//north-west
		f6[row * nx + col][] = tf8[row * nx + col];
		//south-west
		f7[row * nx + col] = tf5[row * nx + col];
		//south-east
		f8[row * nx + col] = tf6[row * nx + col];
	}
}

