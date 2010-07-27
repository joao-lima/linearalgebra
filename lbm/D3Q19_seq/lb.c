#include "lb.h"

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
void alloc_lattice(s_lattice *l) {
	//local variables
	int x, y, z;
	
	//Alloc memory space to the grid 
	//Obstacles matrix
	l->obst = (bool ***) calloc(l->lx, sizeof(bool **));
	for(x = 0; x < l->lx; x++) {
		l->obst[x] = (bool **) calloc(l->ly, sizeof(bool *));
		for(y = 0; y < l->ly; y++) 
			l->obst[x][y] = (bool *) calloc(l->lz, sizeof(bool));
	}
	
	//Lattice and temporary
	l->node = (double ****) calloc(l->lx, sizeof(double***));
	l->temp = (double ****) calloc(l->lx, sizeof(double***));
	for(x = 0; x < l->lx; x++) {
		l->node[x] = (double ***) calloc(l->ly, sizeof(double**));
		l->temp[x] = (double ***) calloc(l->ly, sizeof(double**));
		for(y = 0; y < l->ly; y++) {
			l->node[x][y] = (double **) calloc(l->lz, sizeof(double*));
			l->temp[x][y] = (double **) calloc(l->lz, sizeof(double*));
			for(z = 0; z < l->lz; z++) {
				l->node[x][y][z] = (double *) calloc(l->n, sizeof(double));
				l->temp[x][y][z] = (double *) calloc(l->n, sizeof(double));
			}
		}
	}
}


//////////////////////////////////////////
// Read_obstacles
//////////////////////////////////////////
s_lattice *read_obstacles(string file) { 
	//local variables
	int max;
	int c = 0;
	int i, j, k;
	s_lattice *l = (s_lattice *) malloc(sizeof(s_lattice));
	if (l == NULL)
		printf("Erro\n");
		
	//Open input file
	FILE *archive = fopen(file, "r");
	if (archive == NULL) {
		printf("Could not read colision input file\n\n");
                exit(-2);	
	}

	//Reading headers
	fscanf(archive, "%d", &l->lx);
	fscanf(archive, "%d", &l->ly);
	fscanf(archive, "%d", &l->lz);
	fscanf(archive, "%d", &l->d);
	fscanf(archive, "%d", &l->n);
	fscanf(archive, "%d", &max);

	//printf("%d %d %d %d %d\n", l->lx, l->ly, l->lz, l->n, max);
	
	//alloc memory
	alloc_lattice(l);

	//Reading obstacle points
	while (c < max) {
		fscanf(archive, "%d %d %d", &i, &j, &k);
		//Check if i and j are less then x_max and y max
		//if(i > l->lx || j > l->ly)
		//	printf("Obstacle input file is not valid\n\n");
		//In the file position begin in 1
		l->obst[i][j][k] = true;
		c++;
	}

	//close archive
	fclose(archive);
	
	return l;
}


//////////////////////////////////////////
// Init_constants
//////////////////////////////////////////
void init_constants(s_lattice * l, s_properties *p) {
	//local variables
	int i, j;

	//alloc memory
	l->e = (int **) calloc(l->n, sizeof(int *));
	for(i = 0; i < l->n; i++) 
		l->e[i] = (int *) calloc(l->d, sizeof(int));

	//Lattice vectors for the particles in XY-plane
	l->e[0][0] =  0; l->e[0][1] =  0;  l->e[0][2] = 0;  

	l->e[1][0] =  1; l->e[1][1] =  0;  l->e[1][2] = 0;	  
	l->e[2][0] =  1; l->e[2][1] =  1;  l->e[2][2] = 0;
	l->e[3][0] =  0; l->e[3][1] =  1;  l->e[3][2] = 0;
	l->e[4][0] = -1; l->e[4][1] =  1;  l->e[4][2] = 0;
	l->e[5][0] = -1; l->e[5][1] =  0;  l->e[5][2] = 0;
	l->e[6][0] = -1; l->e[6][1] = -1;  l->e[6][2] = 0;
	l->e[7][0] =  0; l->e[7][1] = -1;  l->e[7][2] = 0;
	l->e[8][0] =  1; l->e[8][1] = -1;  l->e[8][2] = 0;

	// Lattice vectors for the particles in XZ-plane
	l->e[9][0]  =  1; l->e[9][1]  =  0;  l->e[9][2]  =  1;
	l->e[10][0] =  0; l->e[10][1] =  0;  l->e[10][2] =  1;
	l->e[11][0] = -1; l->e[11][1] =  0;  l->e[11][2] =  1;
	l->e[12][0] = -1; l->e[12][1] =  0;  l->e[12][2] = -1;
	l->e[13][0] =  0; l->e[13][1] =  0;  l->e[13][2] = -1;
	l->e[14][0] =  1; l->e[14][1] =  0;  l->e[14][2] = -1;
 
	//Lattice vectors for the particles in YZ-plane

	l->e[15][0] =  0; l->e[15][1] =  1;  l->e[15][2] =  1;
	l->e[16][0] =  0; l->e[16][1] = -1;  l->e[16][2] =  1;
	l->e[17][0] =  0; l->e[17][1] = -1;  l->e[17][2] = -1;
	l->e[18][0] =  0; l->e[18][1] =  1;  l->e[18][2] = -1;

	//Density
	double *t = (double *) calloc(l->d, sizeof(double));
	t[0] = p->density/3;
	t[1] = p->density/18;
	t[2] = p->density/36;

	//The coefficients for eq. distr. A is for the constant term, 
	//B is for e*u-term, C is for (e*u)^2-term and D is for u^2-term 
	l->A = (double *) calloc(l->n, sizeof(double));
	l->B = (double *) calloc(l->n, sizeof(double));
	l->C = (double *) calloc(l->n, sizeof(double));
	l->D = (double *) calloc(l->n, sizeof(double));

	//Coefficients for the particles in 0 point
	l->A[0] = t[0]; l->B[0] = 0.0;      l->C[0] = 0.0;          l->D[0] = -t[0]/2.0/CS2;

	//Coefficients for the particles in XY-plane
	l->A[1] = t[1]; l->B[1] = t[1]/CS2; l->C[1] = t[1]/2.0/CS4; l->D[1] = -t[1]/2.0/CS2;
	l->A[2] = t[2]; l->B[2] = t[2]/CS2; l->C[2] = t[2]/2.0/CS4; l->D[2] = -t[2]/2.0/CS2;
	l->A[3] = t[1]; l->B[3] = t[1]/CS2; l->C[3] = t[1]/2.0/CS4; l->D[3] = -t[1]/2.0/CS2;
	l->A[4] = t[2]; l->B[4] = t[2]/CS2; l->C[4] = t[2]/2.0/CS4; l->D[4] = -t[2]/2.0/CS2;
	l->A[5] = t[1]; l->B[5] = t[1]/CS2; l->C[5] = t[1]/2.0/CS4; l->D[5] = -t[1]/2.0/CS2;
	l->A[6] = t[2]; l->B[6] = t[2]/CS2; l->C[6] = t[2]/2.0/CS4; l->D[6] = -t[2]/2.0/CS2;
	l->A[7] = t[1]; l->B[7] = t[1]/CS2; l->C[7] = t[1]/2.0/CS4; l->D[7] = -t[1]/2.0/CS2;
	l->A[8] = t[2]; l->B[8] = t[2]/CS2; l->C[8] = t[2]/2.0/CS4; l->D[8] = -t[2]/2.0/CS2;

	//Coefficients for the particles in XZ-plane
	l->A[9]  = t[2]; l->B[9]  = t[2]/CS2; l->C[9]  = t[2]/2.0/CS4; l->D[9]  = -t[2]/2.0/CS2;
	l->A[10] = t[1]; l->B[10] = t[1]/CS2; l->C[10] = t[1]/2.0/CS4; l->D[10] = -t[1]/2.0/CS2;
	l->A[11] = t[2]; l->B[11] = t[2]/CS2; l->C[11] = t[2]/2.0/CS4; l->D[11] = -t[2]/2.0/CS2;
	l->A[12] = t[2]; l->B[12] = t[2]/CS2; l->C[12] = t[2]/2.0/CS4; l->D[12] = -t[2]/2.0/CS2;
	l->A[13] = t[1]; l->B[13] = t[1]/CS2; l->C[13] = t[1]/2.0/CS4; l->D[13] = -t[1]/2.0/CS2;
	l->A[14] = t[2]; l->B[14] = t[2]/CS2; l->C[14] = t[2]/2.0/CS4; l->D[14] = -t[2]/2.0/CS2;

	//Coefficients for the particles in YZ-plane
	l->A[15] = t[2]; l->B[15] = t[2]/CS2; l->C[15] = t[2]/2.0/CS4; l->D[15] = -t[2]/2.0/CS2;
	l->A[16] = t[2]; l->B[16] = t[2]/CS2; l->C[16] = t[2]/2.0/CS4; l->D[16] = -t[2]/2.0/CS2;
	l->A[17] = t[2]; l->B[17] = t[2]/CS2; l->C[17] = t[2]/2.0/CS4; l->D[17] = -t[2]/2.0/CS2;
	l->A[18] = t[2]; l->B[18] = t[2]/CS2; l->C[18] = t[2]/2.0/CS4; l->D[18] = -t[2]/2.0/CS2;
	
	//The six tables that tell the numbers of ples that have positive or negative x, y or z component. Each of these tables have 5 (CLNBR) components
	l->pos_x = (int *) calloc(COL, sizeof(int));
	l->neg_x = (int *) calloc(COL, sizeof(int));
	l->pos_y = (int *) calloc(COL, sizeof(int));
	l->neg_y = (int *) calloc(COL, sizeof(int));
	l->pos_z = (int *) calloc(COL, sizeof(int));
	l->neg_z = (int *) calloc(COL, sizeof(int));
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][0] > 0) {
			l->pos_x[i] = j;
			i++;
		}
	}
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][0] < 0) {
			l->neg_x[i] = j;
			i++;
		}
	}
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][1] > 0) {
			l->pos_y[i] = j;
			i++;
		}
	}
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][1] < 0) {
			l->neg_y[i] = j;
			i++;
		}
	}
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][2] > 0) {
			l->pos_z[i] = j;
			i++;
		}
	}
	
	i = 0;
	for(j=1; j<NDIM; j++) {
		if(l->e[j][2] < 0) {
			l->neg_z[i] = j;
			i++;
		}
	}
	
	l->points = (int ***) calloc(l->d, sizeof(int **));
	for(i = 0; i < l->d; i++) {
		l->points[i] = (int **) calloc(l->d, sizeof(int *));
		for(j = 0; j < l->d; j++) 
			l->points[i][j] = (int *) calloc(l->d, sizeof(int));
	}

	//Tables that give the population numbers when the direction is known
	//0 = -component, 1 = no component 2 = +component
	l->points[0+1][0+1][0+1]   = 0;  l->points[1+1][0+1][0+1]   = 1;
	l->points[1+1][1+1][0+1]   = 2;  l->points[0+1][1+1][0+1]   = 3; 
	l->points[-1+1][1+1][0+1]  = 4;  l->points[-1+1][0+1][0+1]  = 5; 
	l->points[-1+1][-1+1][0+1] = 6;  l->points[0+1][-1+1][0+1]  = 7; 
	l->points[1+1][-1+1][0+1]  = 8;  l->points[1+1][0+1][1+1]   = 9;
	l->points[0+1][0+1][1+1]   = 10; l->points[-1+1][0+1][1+1]  = 11; 
	l->points[-1+1][0+1][-1+1] = 12; l->points[0+1][0+1][-1+1]  = 13; 
	l->points[1+1][0+1][-1+1]  = 14; l->points[0+1][1+1][1+1]   = 15; 
	l->points[0+1][-1+1][1+1]  = 16; l->points[0+1][-1+1][-1+1] = 17; 
	l->points[0+1][1+1][-1+1]  = 18; 
}


//////////////////////////////////////////
// Init_density
//////////////////////////////////////////
void init_density(s_lattice * l) {
	//local variables
	int x, y, z, n;
	
	//loop over computational domain
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (z = 0; z < l->lz; z++) {
				for (n = 0; n < l->n; n++) {
					l->node[x][y][z][n] = l->A[n];
				}
			}
		}
	}
}


//////////////////////////////////////////
// Check_density
//////////////////////////////////////////
void check_density(s_lattice *l, int time) {
	//local variables
	int x, y, z, n;
	double n_sum = 0;
	
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (z = 0; z < l->lz; z++) {
				for (n = 0; n < l->n; n++) {
					n_sum += l->node[x][y][z][n];
				} //printf("%4.3f ", n_sum);
			} //printf("\n");
		} //printf("\n");
	}
	
	printf("Iteration number = %d ", time);
	printf("Integral density = %f\n", n_sum);
}


//////////////////////////////////////////
// Redistribute
//////////////////////////////////////////
// It is interesting to redistribute de forces to all points
void redistribute(s_lattice *l, double accel, double density) {
	//local variables
	int x, y, z, n;
	double t_1 = density * accel / 18.0;
	double t_2 = density * accel / 36.0;
	int v_1[NDIM], v_2[NDIM];
	int length = 1;

	v_1[0] = 1;
	v_2[0] = 5;
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (z = 0; z < l->lz; z++) {
				//check to avoid negative densities
				//check false | true
				if (l->obst[x][y][z] == false) {
					for (n = 0; n < length; n++) {
						l->node[x][y][z][v_1[n]] += t_1;
						l->node[x][y][z][v_2[n]] -= t_1;
//						if (l->node[x][y][z][v_2[0]] < 0)
//							printf("%d %d %d %d\n", x, y, z, v_2[0]);
					}
				}
			}
		}
	}

	v_1[0] = 2;
	v_1[1] = 8;
	v_1[2] = 9;
	v_1[3] = 14;

	v_2[0] = 4;
	v_2[1] = 6;
	v_2[2] = 11;
	v_2[3] = 12;
	
	length = 4;
	
	for (x = 0; x < l->lx; x++) {
		for (y = 0; y < l->ly; y++) {
			for (z = 0; z < l->lz; z++) {
				//check to avoid negative densities
				//check false | true
				if (l->obst[x][y][z] == false) {
					for (n = 0; n < length; n++) {
						l->node[x][y][z][v_1[n]] += t_2;
						l->node[x][y][z][v_2[n]] -= t_2;
//						if (l->node[x][y][z][v_2[0]] < 0)
//							printf("%d %d %d %d\n", x, y, z, v_2[0]);
					}
				}				
			}
		}
	}
}


//////////////////////////////////////////
// Propagate
//////////////////////////////////////////
void propagate(s_lattice *l) {
        //local variables
	int x, y, z;
	int x_e = 0, x_w = 0, y_u = 0, y_d = 0, z_n = 0, z_s = 0;
	
	for (x = 0; x < l->lx; x++) {
		for(y = 0; y < l->ly; y++) {
			for(z = 0; z < l->lz; z++) {
			
				//compute upper and right next neighbour nodes
				x_e = (x + 1)%l->lx;
				y_u = (y + 1)%l->ly;
				z_n = (z + 1)%l->lz;
			
				//compute lower and left next neighbour nodes
				x_w = (x - 1 + l->lx)%l->lx;
				y_d = (y - 1 + l->ly)%l->ly;
				z_s = (z - 1 + l->lz)%l->lz;
				//density propagation
				
				//zero
				l->temp[x][y][z][0] = l->node[x][y][z][0];

				l->temp[x_e][y][z][1] = l->node[x][y][z][1];
				l->temp[x_e][y_u][z][2] = l->node[x][y][z][2];
				l->temp[x][y_u][z][3] = l->node[x][y][z][3];
				l->temp[x_w][y_u][z][4] = l->node[x][y][z][4];
				l->temp[x_w][y][z][5] = l->node[x][y][z][5];
				l->temp[x_w][y_d][z][6] = l->node[x][y][z][6];
				l->temp[x][y_d][z][7] = l->node[x][y][z][7];
				l->temp[x_e][y_d][z][8] = l->node[x][y][z][8];

				l->temp[x_e][y][z_n][9] = l->node[x][y][z][9];
				l->temp[x][y][z_n][10] = l->node[x][y][z][10];
				l->temp[x_w][y][z_n][11] = l->node[x][y][z][11];
				l->temp[x_w][y][z_s][12] = l->node[x][y][z][12];
				l->temp[x][y][z_s][13] = l->node[x][y][z][13];
				l->temp[x_e][y][z_s][14] = l->node[x][y][z][14];

				l->temp[x][y_u][z_n][15] = l->node[x][y][z][15];
				l->temp[x][y_d][z_n][16] = l->node[x][y][z][16];
				l->temp[x][y_d][z_s][17] = l->node[x][y][z][17];
				l->temp[x][y_u][z_s][18] = l->node[x][y][z][18];
			}
		}
	}
}


//////////////////////////////////////////
// Bounceback
//////////////////////////////////////////
void bounceback(s_lattice *l) {
	//local variables
	int x, y, z;

	for (x = 0; x < l->lx; x++) {
		for(y = 0; y < l->ly; y++) {
			for(z = 0; z < l->lz; z++) {
				if (l->obst[x][y][z] == true) {
					l->node[x][y][z][1] = l->temp[x][y][z][5];
					l->node[x][y][z][2] = l->temp[x][y][z][6];
					l->node[x][y][z][3] = l->temp[x][y][z][7];
					l->node[x][y][z][4] = l->temp[x][y][z][8];
					l->node[x][y][z][5] = l->temp[x][y][z][1];
					l->node[x][y][z][6] = l->temp[x][y][z][2];
					l->node[x][y][z][7] = l->temp[x][y][z][3];
					l->node[x][y][z][8] = l->temp[x][y][z][4];

					l->node[x][y][z][9] = l->temp[x][y][z][12];
					l->node[x][y][z][10] = l->temp[x][y][z][13];
					l->node[x][y][z][11] = l->temp[x][y][z][14];
					l->node[x][y][z][12] = l->temp[x][y][z][9];
					l->node[x][y][z][13] = l->temp[x][y][z][10];
					l->node[x][y][z][14] = l->temp[x][y][z][11];

					l->node[x][y][z][15] = l->temp[x][y][z][17];
					l->node[x][y][z][16] = l->temp[x][y][z][18];
					l->node[x][y][z][17] = l->temp[x][y][z][15];
					l->node[x][y][z][18] = l->temp[x][y][z][16];
				}
			}
		}
	}
}


//////////////////////////////////////////
// Relaxation
//////////////////////////////////////////
void relaxation(s_lattice *l, double density, double omega) {
	//local variables
	int x, y, z, i;
	double c_squ = 1.0 / 3.0;
	double t_0 = 1.0 / 3.0;
	double t_1 = 1.0 / 18.0;
	double t_2 = 1.0 / 36.0;
	double u_x, u_y, u_z;
	double u_n[l->n], n_equ[l->n], u_squ, d_loc;

	for (x = 0; x < l->lx; x++) {
		for(y = 0; y < l->ly; y++) {
			for(z = 0; z < l->lz; z++) {
				if (l->obst[x][y][z] == false) {
					d_loc = 0.0;
					for (i = 0; i < l->n; i++) {
						d_loc += l->temp[x][y][z][i];
					}

					//x-, y- and z- velocity components
					u_x = (l->temp[x][y][z][1] + l->temp[x][y][z][2] + l->temp[x][y][z][8] + l->temp[x][y][z][9] + l->temp[x][y][z][14]  - (l->temp[x][y][z][4] + l->temp[x][y][z][5] + l->temp[x][y][z][6] + l->temp[x][y][z][11] + l->temp[x][y][z][12])) / d_loc;

					u_y = (l->temp[x][y][z][2] + l->temp[x][y][z][3] + l->temp[x][y][z][4] + l->temp[x][y][z][15] + l->temp[x][y][z][18]  - (l->temp[x][y][z][6] + l->temp[x][y][z][7] + l->temp[x][y][z][8] + l->temp[x][y][z][16] + l->temp[x][y][z][17])) / d_loc;
					
					u_z = (l->temp[x][y][z][9] + l->temp[x][y][z][10] + l->temp[x][y][z][11] + l->temp[x][y][z][15] + l->temp[x][y][z][16]  - (l->temp[x][y][z][12] + l->temp[x][y][z][13] + l->temp[x][y][z][14] + l->temp[x][y][z][17] + l->temp[x][y][z][18])) / d_loc;
	
					//square velocity
					u_squ = u_x * u_x + u_y * u_y + u_z * u_z;
					//printf("%f %f %f %f\n", u_x, u_y, u_z, u_squ);
					//n- velocity compnents
					u_n[1] = u_x;
					u_n[2] = u_x + u_y;
					u_n[3] = u_y;
					u_n[4] = -u_x + u_y;
					u_n[5] = -u_x;
					u_n[6] = -u_x - u_y;
					u_n[7] = -u_y;
					u_n[8] = u_x - u_y;

					u_n[9] = u_x + u_z;
					u_n[10] = u_z;
					u_n[11] = -u_x + u_z;
					u_n[12] = -u_x - u_z;
					u_n[13] = -u_z;
					u_n[14] = u_x - u_z;
					u_n[15] = u_y + u_z;
					u_n[16] = -u_y + u_z;
					u_n[17] = -u_y - u_z;
					u_n[18] = u_y - u_z;
							  
					//zero velocity density
					n_equ[0] = t_0 * d_loc * (1.0 - u_squ / (2.0 * c_squ));
					//axis speeds: factor: t_1
					n_equ[1] = t_1 * d_loc * (1.0 + u_n[1] / c_squ + u_n[1] * u_n[1] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[2] = t_2 * d_loc * (1.0 + u_n[2] / c_squ + u_n[2] * u_n[2] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[3] = t_1 * d_loc * (1.0 + u_n[3] / c_squ + u_n[3] * u_n[3] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[4] = t_2 * d_loc * (1.0 + u_n[4] / c_squ + u_n[4] * u_n[4] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[5] = t_1 * d_loc * (1.0 + u_n[5] / c_squ + u_n[5] * u_n[5] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[6] = t_2 * d_loc * (1.0 + u_n[6] / c_squ + u_n[6] * u_n[6] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					
					//diagonal speeds: factor t_2
					n_equ[7] = t_1 * d_loc * (1.0 + u_n[7] / c_squ + u_n[7] * u_n[7] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[8] = t_2 * d_loc * (1.0 + u_n[8] / c_squ + u_n[8] * u_n[8] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[9] = t_2 * d_loc * (1.0 + u_n[9] / c_squ + u_n[9] * u_n[9] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[10] = t_1* d_loc * (1.0 + u_n[10] / c_squ + u_n[10] * u_n[10] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[11] = t_2 * d_loc * (1.0 + u_n[11] / c_squ + u_n[11] * u_n[11] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[12] = t_2 * d_loc * (1.0 + u_n[12] / c_squ + u_n[12] * u_n[12] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[13] = t_1 * d_loc * (1.0 + u_n[13] / c_squ + u_n[13] * u_n[13] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[14] = t_2 * d_loc * (1.0 + u_n[14] / c_squ + u_n[14] * u_n[14] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[15] = t_2 * d_loc * (1.0 + u_n[15] / c_squ + u_n[15] * u_n[15] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[16] = t_2 * d_loc * (1.0 + u_n[16] / c_squ + u_n[16] * u_n[16] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[17] = t_2 * d_loc * (1.0 + u_n[17] / c_squ + u_n[17] * u_n[17] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
					n_equ[18] = t_2 * d_loc * (1.0 + u_n[18] / c_squ + u_n[18] * u_n[18] / (2.0 * c_squ * c_squ) - u_squ / (2.0 * c_squ));
				
					//relaxation step
					for (i = 0; i < l->n; i++) {
						l->node[x][y][z][i] = l->temp[x][y][z][i] + omega * (n_equ[i] - l->temp[x][y][z][i]);
					}	
				}
			}
		}
	}
}


//////////////////////////////////////////
// Calc_velocity
//////////////////////////////////////////
double calc_velocity(s_lattice *l, int time) {
	//local variables
	int x, y, z, i, n_free;
	double u_x, d_loc;

	x = l->lx/2;
	n_free = 0;
	u_x = 0;

	for(y = 0; y < l->ly; y++) {
		for(z = 0; z < l->lz; z++) {
			if (l->obst[x][y][z] == false) {
				d_loc = 0;	
				for (i = 0; i < l->n; i++)
					d_loc = d_loc + l->node[x][y][z][i];
				u_x = u_x + (l->node[x][y][z][1] + l->node[x][y][z][2] + l->node[x][y][z][8] + l->node[x][y][z][9] + l->node[x][y][z][14]  - (l->node[x][y][z][4] + l->node[x][y][z][5] + l->node[x][y][z][6] + l->node[x][y][z][11] + l->node[x][y][z][12])) / d_loc;
				n_free++;
			}
		}
	}
	
	//Optional
	printf("%d %lf\n", time, u_x / n_free);
	return u_x / n_free;
}


////////////////////////////////////////////
//// Write_results
////////////////////////////////////////////
void write_results(string file, s_lattice *l, double density) {
	//local variables
	int x, y, z, i;
	bool obsval;
	double u_x, u_y, u_z, d_loc, press;

	//Square speed of sound
	double c_squ = 1.0 / 3.0;

	//Open results output file
	FILE *archive = fopen(file, "w");

	//write results
	fprintf(archive,"VARIABLES = X, Y, Z, VX, VY, VZ, PRESS, OBST\n");
	fprintf(archive,"ZONE I= %d, J= %d, K= %d F=POINT\n", l->lx, l->ly, l->lz);

	for(z = 0; z < l->lz; z++) {
		for(y = 0; y < l->ly; y++) {
			for(x = 0; x < l->lx; x++) {
				//if obstacle node, nothing is to do
				if (l->obst[x][y][z] == true) {
					//obstacle indicator
					obsval = true;
					//velocity components = 0
					u_x = 0.0;
					u_y = 0.0;
					u_z = 0.0;
					//pressure = average pressure
					press = density * c_squ;
				} else {
					//integral local density
					//initialize variable d_loc
					d_loc = 0.0;
					for (i = 0; i < l->n; i++) {
						d_loc += l->node[x][y][z][i];
					}
					
					//x-, y- and z- velocity components
					u_x = l->node[x][y][z][1] + l->node[x][y][z][2] + l->node[x][y][z][8] + l->node[x][y][z][9] + l->node[x][y][z][14]  - (l->node[x][y][z][4] + l->node[x][y][z][5] + l->node[x][y][z][6] + l->node[x][y][z][11] + l->node[x][y][z][13]);
					u_x /= d_loc;

					u_y = l->node[x][y][z][2] + l->node[x][y][z][3] + l->node[x][y][z][4] + l->node[x][y][z][15] + l->node[x][y][z][18]  - (l->node[x][y][z][6] + l->node[x][y][z][7] + l->node[x][y][z][8] + l->node[x][y][z][16] + l->node[x][y][z][17]);
					u_y /= d_loc;
					
					u_z = l->node[x][y][z][9] + l->node[x][y][z][10] + l->node[x][y][z][11] + l->node[x][y][z][15] + l->node[x][y][z][16]  - (l->node[x][y][z][12] + l->node[x][y][z][13] + l->node[x][y][z][14] + l->node[x][y][z][17] + l->node[x][y][z][18]);
					u_z /= d_loc;
	
					//pressure
					press = d_loc * c_squ;
					obsval = false;
				}
				fprintf(archive,"%d %d %d %f %f %f %f %d\n", x, y, z, u_x, u_y, u_z, press, obsval);
			}
		}
	}
	//close the file
	fclose(archive);
}


//////////////////////////////////////////
// Compute Reynolds number
//////////////////////////////////////////
void comp_rey(s_lattice *l, s_properties *p, int time, double execution) {

	//Local variables
	double vel, visc, rey;

	//Compute average velocity
	vel = calc_velocity(l, time);
	
	//Compute viscosity
	visc = 1.0 / 6.0 * (2.0 / p->omega - 1.0);
	
	//Compute Reynolds number
	rey = vel * p->r_rey / visc;
	
	FILE *t = fopen("time.out", "a");
	FILE *e = fopen("exec.out", "a");

	fprintf(t, "%d %lg %lg %lg\n", time, visc, vel, rey);
	fprintf(e, "%lg\n", execution);
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
