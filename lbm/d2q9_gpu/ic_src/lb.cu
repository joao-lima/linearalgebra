#include "lb.h"

/*__global__ void GPUmain_loop(s_lattice *l, s_properties *p, int time){

  if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
    GPUredistribute(l, p->accel, p->density);
  }
  __syncthreads();

  GPUpropagate(l);
  __syncthreads();

  GPUbounceback(l);
  __syncthreads();

  GPUrelaxation(l, p->density, p->omega);
  __syncthreads();

  if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
    p->vel = GPUcalc_velocity(l, time);
  }
  __syncthreads();

}*/

void init_density(s_lattice * l, double density) {
        //local variables
        int x, y;
        double t_0 = density * 4.0 / 9.0;
        double t_1 = density / 9.0;
        double t_2 = density / 36.0;
        l->n_sum = 0.0;

        // calculate X and Y
        //x = blockIdx.x * blockDim.x + threadIdx.x;
        //y = blockIdx.y * blockDim.y + threadIdx.y;

        //loop over computational domain
        for (x = 0; x < l->lx; x++) {
                for (y = 0; y < l->ly; y++) {
                //if ((x < l->lx) && (y < l->ly)){
                        //printf("%d,%d\n",x,y);
                        //zero velocity density
                        l->node[x][y][0] = t_0;
                        //equilibrium densities for axis speeds
                        l->node[x][y][1] = t_1;
                        l->node[x][y][2] = t_1;
                        l->node[x][y][3] = t_1;
                        l->node[x][y][4] = t_1;
                        //equilibrium densities for diagonal speeds
                        l->node[x][y][5] = t_2;
                        l->node[x][y][6] = t_2;
                        l->node[x][y][7] = t_2;
                        l->node[x][y][8] = t_2;
                //}
                }
        }
}



//////////////////////////////////////////
// Check_density
//////////////////////////////////////////
__global__ void GPUcheck_density(s_lattice *l, double *sum) {
	//local variables
	int x, y, n;
	l->n_sum = 0.0;
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (n = 0; n < l->n; n++) {
				l->n_sum += l->node[x][y][n];
			}
		}
	}
	*sum = l->n_sum;
}



//////////////////////////////////////////
// Compute Reynolds number
//////////////////////////////////////////
__global__ void GPUcomp_rey(s_lattice *l, s_properties *p, int time){
	int x, y, i, n_free;
	double u_x, d_loc;

	x = l->lx/2;
	n_free = 0;
	u_x = 0;

	for(y = 0; y < l->ly; y++) {
		if (l->obst[x][y] == false) {
			d_loc = 0;
			for (i = 0; i < l->n; i++)
				d_loc = d_loc + l->node[x][y][i];
			u_x = u_x + (l->node[x][y][1] + l->node[x][y][5] + l->node[x][y][8] - (l->node[x][y][3] + l->node[x][y][6] + l->node[x][y][7])) / d_loc;
			n_free++;
		}
	}

	//Compute average velocity
	p->vel = u_x / n_free;
	
	//Compute viscosity
	p->visc = 1.0 / 6.0 * (2.0 / p->omega - 1.0);
	
	//Compute Reynolds number
	p->rey = p->vel * p->r_rey / p->visc;

}


//////////////////////////////////////////
// Calc_velocity
//////////////////////////////////////////
__global__ void GPUcalc_velocity(s_lattice *l, int time, double *vel) {
//  if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){

  //local variable
  int x, y, i, n_free;
  double u_x, d_loc;

  x = l->lx/2;
  n_free = 0;
  u_x = 0;

  for(y = 0; y < l->ly; y++) {
    if (l->obst[x][y] == false) {
      d_loc = 0;
      for (i = 0; i < l->n; i++)
        d_loc = d_loc + l->node[x][y][i];
        u_x = u_x + (l->node[x][y][1] + l->node[x][y][5] + l->node[x][y][8] - (l->node[x][y][3] + l->node[x][y][6] + l->node[x][y][7])) / d_loc;
        n_free++;
    }
  }

  *vel = u_x / n_free;
  //return (u_x / n_free);

//  }
}


//////////////////////////////////////////
// Redistribute
//////////////////////////////////////////
__global__ void GPUredistribute(s_lattice *l, double accel, double density){

//  if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
	//local variables
	int y;//, begin, end;
	double t_1 = density * accel / 9.0;
	double t_2 = density * accel / 36.0;

	for (y = 0; y < l->ly; y++) {
		//check to avoid negative densities
		//check false | true
		if ((l->obst[0][y] == false) && (l->node[0][y][3] - t_1 > 0) && (l->node[0][y][6] - t_2 > 0) && (l->node[0][y][7] - t_2 > 0)) {
			//increase east
			l->node[0][y][1] += t_1;
			//decrease west
			l->node[0][y][3] -= t_1;
			//increase north-east
			l->node[0][y][5] += t_2;
			//decrease north-west
			l->node[0][y][6] -= t_2;
			//decrease south-west
			l->node[0][y][7] -= t_2;
			//increase south-east
			l->node[0][y][8] += t_2;
		}
	}
//  }
}

//////////////////////////////////////////
// Propagate
//////////////////////////////////////////
__global__ void GPUpropagate(s_lattice *l){
// if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
        //local variables
	int x, y;
	int x_e = 0, x_w = 0, y_n = 0, y_s = 0;

	/*x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;*/

	int blockRangeX = (l->lx / gridDim.x);
	int threadRangeX = (blockRangeX / blockDim.x);

	int blockBeginX = blockRangeX * blockIdx.x;
	//int blockEndX = blockBeginX + blockRangeX;

	int threadBeginX = blockBeginX + threadRangeX * threadIdx.x;
	int threadEndX = threadBeginX + threadRangeX;

	int blockRangeY = (l->ly / gridDim.y);
	int threadRangeY = (blockRangeY / blockDim.y);

	int blockBeginY = blockRangeY * blockIdx.y;
	//int blockEndY = blockBeginY + blockRangeY;

	int threadBeginY = blockBeginY + threadRangeY * threadIdx.y;
	int threadEndY = threadBeginY + threadRangeY;

	/*if ((blockIdx.x == 0)&&(blockIdx.y == 0)) {
	  printf("threadIdx.x (%d) threadIdx.y (%d)\n", threadIdx.x, threadIdx.y);
	  printf("threadBeginX (%d) threadEndX (%d)\n", threadBeginX, threadEndX);
	  printf("threadBeginY (%d) threadEndY (%d)\n", threadBeginY, threadEndY);
	  //printf("yBegin (%d) yEnd (%d)\n", yBegin, yEnd);
	}*/
	
	//if ((x < l->lx) && (y < l->ly)){
//	for (x = threadBeginX; x < threadEndX; x++) {
//		for(y = threadBeginY; y < threadEndY; y++) {
        for (x = 0; x < l->lx; x++) {
                for(y = 0; y < l->ly; y++) {
			//compute upper and right next neighbour nodes
			x_e = (x + 1)%l->lx;
			y_n = (y + 1)%l->ly;
			
			//compute lower and left next neighbour nodes
			x_w = (x - 1 + l->lx)%l->lx;
			y_s = (y - 1 + l->ly)%l->ly;
			
			//density propagation
			
			//zero
			l->temp[x][y][0] = l->node[x][y][0];
			//east
			l->temp[x_e][y][1] = l->node[x][y][1];
			//north
			l->temp[x][y_n][2] = l->node[x][y][2];
			//west
			l->temp[x_w][y][3] = l->node[x][y][3];
			//south
			l->temp[x][y_s][4] = l->node[x][y][4];
			//north-east
			l->temp[x_e][y_n][5] = l->node[x][y][5];
			//north-west
			l->temp[x_w][y_n][6] = l->node[x][y][6];
			//south-west
			l->temp[x_w][y_s][7] = l->node[x][y][7];
			//south-east
			l->temp[x_e][y_s][8] = l->node[x][y][8];
		}
	}
//  }
}

//////////////////////////////////////////
// Bounceback
//////////////////////////////////////////
__global__ void GPUbounceback(s_lattice *l) {
// if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){

	//local variables
	int x, y;

	/*x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;*/

	int blockRangeX = (l->lx / gridDim.x);
	int threadRangeX = (blockRangeX / blockDim.x);

	int blockBeginX = blockRangeX * blockIdx.x;
	//int blockEndX = blockBeginX + blockRangeX;

	int threadBeginX = blockBeginX + threadRangeX * threadIdx.x;
	int threadEndX = threadBeginX + threadRangeX;

	int blockRangeY = (l->ly / gridDim.y);
	int threadRangeY = (blockRangeY / blockDim.y);

	int blockBeginY = blockRangeY * blockIdx.y;
	//int blockEndY = blockBeginY + blockRangeY;

	int threadBeginY = blockBeginY + threadRangeY * threadIdx.y;
	int threadEndY = threadBeginY + threadRangeY;
	
	//if ((x < l->lx) && (y < l->ly)){
//	for (x = threadBeginX; x < threadEndX; x++) {
//		for(y = threadBeginY; y < threadEndY; y++) {
        for (x = 0; x < l->lx; x++) {
                for(y = 0; y < l->ly; y++) {
			if (l->obst[x][y] == true) {
				//east
				l->node[x][y][1] = l->temp[x][y][3];
				//north
				l->node[x][y][2] = l->temp[x][y][4];
				//west
				l->node[x][y][3] = l->temp[x][y][1];
				//south
				l->node[x][y][4] = l->temp[x][y][2];
				//north-east
				l->node[x][y][5] = l->temp[x][y][7];
				//north-west
				l->node[x][y][6] = l->temp[x][y][8];
				//south-west
				l->node[x][y][7] = l->temp[x][y][5];
				//south-east
				l->node[x][y][8] = l->temp[x][y][6];
			}
		}
	}
//  }
}

//////////////////////////////////////////
// Relaxation
//////////////////////////////////////////
__global__ void GPUrelaxation(s_lattice *l, double density, double omega) {
// if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
	//local variables
	int x, y, i;
	double c_squ = 1.0 / 3.0;
	double t_0 = 4.0 / 9.0;
	double t_1 = 1.0 / 9.0;
	double t_2 = 1.0 / 36.0;
	double u_x, u_y;
	double u_n[9], n_equ[9], u_squ, d_loc;

	/*x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;*/

	int blockRangeX = (l->lx / gridDim.x);
	int threadRangeX = (blockRangeX / blockDim.x);

	int blockBeginX = blockRangeX * blockIdx.x;
	//int blockEndX = blockBeginX + blockRangeX;

	int threadBeginX = blockBeginX + threadRangeX * threadIdx.x;
	int threadEndX = threadBeginX + threadRangeX;

	int blockRangeY = (l->ly / gridDim.y);
	int threadRangeY = (blockRangeY / blockDim.y);

	int blockBeginY = blockRangeY * blockIdx.y;
	//int blockEndY = blockBeginY + blockRangeY;

	int threadBeginY = blockBeginY + threadRangeY * threadIdx.y;
	int threadEndY = threadBeginY + threadRangeY;
	
	//if ((x < l->lx) && (y < l->ly)){
//	for (x = threadBeginX; x < threadEndX; x++) {
//		for(y = threadBeginY; y < threadEndY; y++) {
        for (x = 0; x < l->lx; x++) {
                for(y = 0; y < l->ly; y++) {

			if (l->obst[x][y] == false) {
				d_loc = 0.0;
				for (i = 0; i < l->n; i++) {
					d_loc += l->temp[x][y][i];
				}

				//x-, and y- velocity components
				u_x = (l->temp[x][y][1] + l->temp[x][y][5] + l->temp[x][y][8] - (l->temp[x][y][3] + l->temp[x][y][6] + l->temp[x][y][7])) / d_loc;
				u_y = (l->temp[x][y][2] + l->temp[x][y][5] + l->temp[x][y][6] - (l->temp[x][y][4] + l->temp[x][y][7] + l->temp[x][y][8])) / d_loc;

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
				for (i = 0; i < l->n; i++) {
					l->node[x][y][i] = l->temp[x][y][i] + omega * (n_equ[i] - l->temp[x][y][i]);
				}	
			}
		}
	}
//  }
}


















//////////////////////////////////////////
// Functions
//////////////////////////////////////////

//////////////////////////////////////////
// Read_parametrs
//////////////////////////////////////////
s_properties *read_parametrs(string file) {
	//struct with macroscopic properties of a fluid
	s_properties *p = (s_properties*) malloc(sizeof(s_properties));

	//Open input file
	FILE *archive = fopen(file, "r");
	//Checking if the file is correct
	if (archive == NULL) {
		printf("Could not open input file\n\n");
		exit(-2);
	}
	//number of iterations
	fscanf(archive, "%d", &p->t_max);
	//fluid density per link
	fscanf(archive, "%lf", &p->density);
	//density redistribution
	fscanf(archive, "%lf", &p->accel);
	//relaxation parameter
	fscanf(archive, "%lf", &p->omega);
	//linear dimension (for reynolds number)
	fscanf(archive, "%lf", &p->r_rey);
	
	//Close file
	fclose(archive);

	//printf("Parameters read sucessful\n"); //All right
	return p;
}


//////////////////////////////////////////
// Alloc memory to the lattices structures
//////////////////////////////////////////
/*void alloc_lattice(s_lattice *l) {
	//local variables
	int x, y;

	//Alloc memory space to the grid 
	//Obstacles matrix
	l->obst = (int **) calloc(l->lx, sizeof(int *));
	for(x = 0; x < l->lx; x++) 
		l->obst[x] = (int *) calloc(l->ly, sizeof(int));

	//Lattice and temporary
	l->node = (double ***) calloc(l->lx, sizeof(double**));
	l->temp = (double ***) calloc(l->lx, sizeof(double**));
	for(x = 0; x < l->lx; x++) {
		l->node[x] = (double **) calloc(l->ly, sizeof(double*));
		l->temp[x] = (double **) calloc(l->ly, sizeof(double*));
		for(y = 0; y < l->ly; y++) {
			l->node[x][y] = (double *) calloc(l->n, sizeof(double));
			l->temp[x][y] = (double *) calloc(l->n, sizeof(double));
		}
	}
}*/


//////////////////////////////////////////
// Read_obstacles
//////////////////////////////////////////
s_lattice *read_obstacles(string file) { 
	//local variables
	int max;
	int c = 0;
	int i, j;
	s_lattice *l = (s_lattice *) calloc(1, sizeof(s_lattice));
		
	//Open input file
	FILE *archive = fopen(file, "r");
	if (archive == NULL) {
		printf("Could not read colision input file\n\n");
                exit(-2);	
	}

	//Reading headers
	fscanf(archive, "%d", &l->lx);
	fscanf(archive, "%d", &l->ly);
	fscanf(archive, "%d", &l->n);
	fscanf(archive, "%d", &max);

	//printf("%d %d %d %d\n", l->lx, l->ly, l->n, max);
	
	//alloc memory
//	alloc_lattice(l);

	//Reading obstacle points
	while (c < max) {
		fscanf(archive, "%d %d", &i, &j);
		//Check if i and j are less then x_max and y max
		//if(i > l->lx || j > l->ly)
		//	printf("Obstacle input file is not valid\n\n");
		//If the file position begin in 1
		//l->obst[i - 1][j - 1] = true;
		l->obst[i][j] = true;
		c++;
	}

	//close archive
	fclose(archive);
	return l;
}



//////////////////////////////////////////
// Check_density
//////////////////////////////////////////
void check_density(s_lattice *l, int time) {
	//local variables
	int x, y, n;
	double n_sum;
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (n = 0; n < l->n; n++) {
				n_sum = n_sum + l->node[x][y][n];
			}
		}
	}
	printf("Iteration number = %d\n", time);
	printf("Integral density = %f\n", n_sum);
}




////////////////////////////////////////////
//// Write_results
////////////////////////////////////////////
void write_results(string file, s_lattice *l, s_properties *p, int time, double execution, int griddim, int blockdim) {
//
//
	//local variables
	int x, y, i;
	int obsval;
	double u_x, u_y, d_loc, press;

	//Square speed of sound
	double c_squ = 1.0 / 3.0;

	//Open results output file
	FILE *archive = fopen(file, "w");

	//write results
	fprintf(archive,"VARIABLES = X, Y, VX, VY, PRESS, OBST\n");
	fprintf(archive,"ZONE I= %d, J= %d, F=POINT\n", l->lx, l->ly);

	for(y = 0; y < l->ly; y++) {
		for(x = 0; x < l->lx; x++) {
			//if obstacle node, nothing is to do
			if (l->obst[x][y] == true) {
				//obstacle indicator
				obsval = true;
				//velocity components = 0
				u_x = 0.0;
				u_y = 0.0;
				//pressure = average pressure
				press = p->density * c_squ;
			} else {
				//integral local density
				//initialize variable d_loc
				d_loc = 0.0;
				for (i = 0; i < 9; i++) {
					d_loc += l->node[x][y][i];
				}
				// x-, and y- velocity components
				u_x = (l->node[x][y][1] + l->node[x][y][5] + l->node[x][y][8] - (l->node[x][y][3] + l->node[x][y][6] + l->node[x][y][7])) / d_loc;

				u_y = (l->node[x][y][2] + l->node[x][y][5] + l->node[x][y][6] - (l->node[x][y][4] + l->node[x][y][7] + l->node[x][y][8])) / d_loc;
				
				//pressure
				press = d_loc * c_squ;
				obsval = false;
			}
			fprintf(archive,"%d %d %f %f %f %d\n", x, y, u_x, u_y, press, obsval);
		}
	}
	
	fclose(archive);

	FILE *t = fopen("time.out", "a");
	FILE *e = fopen("exec.out", "a");

	fprintf(t, "%d %lg %lg %lg\n", time, p->visc, p->vel, p->rey);
	fprintf(e, "%lg %d %d\n", execution, griddim, blockdim);
	fclose(t);
	fclose(e);

}

//////////////////////////////////////////
// Return de time in a especific moment
//////////////////////////////////////////
double crono() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

