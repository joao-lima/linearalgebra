///////////////////////////////////////////////
//libraries
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef enum {false, true} bool;
typedef char *string;

// The dimension of the lattice space
#define 	 DIM 		 	3

// The number of different links: D3Q19 Model
#define 	 NDIM 			19

//The number of ples that have positive or neg. component to a given direction
#define		 COL			5

// The sound velocity = 1/sqrt(3)
#define 	 CS 		 	0.5773502692

// The sound velocity squared
#define 	 CS2 		 	(1.0/3.0)
#define          CS4             	(CS2*CS2)

// The eq. coeff. has been copied from Y.H. Qian et al. Europhys. Lett. 17 (6) 479
#define  	 t0		 	(1.0/3.0)
#define  	 t1 		 	(1.0/18.0)
#define		 t2		 	(1.0/36.0)


///////////////////////////////////////////////
//structs

//struct contends macroscopic information
typedef struct{
	int t_max; //Maximum number of iterations
	double density; //Fluid density per link
	double accel; //Accelleration
	double omega; //Relaxation parameter
	double r_rey; //Linear dimension for Reynolds number
} s_properties;

//lattice structure
typedef struct {
	int lx; //nodes number in axis x
	int ly; //nodes number in axis y
	int lz; //nodes number in axis z
	int d; //spatial dimensions
	int n; //lattice dimension elements
	bool ***obst; //Obstacle Array lx * ly * lz
	double ****node; //n-speed lattice  lx * ly * lz * n
	double ****temp; //temporarely storage of fluid densities
	int **e; //edges
	double *A, *B, *C, *D;
	int *pos_x, *pos_y, *pos_z;
	int *neg_x, *neg_y, *neg_z;
	int ***points;
} s_lattice;


///////////////////////////////////////////////
//Functions

s_properties* read_parametrs(string file);

void alloc_lattice(s_lattice *l);

s_lattice* read_obstacles(string file);

void init_constants(s_lattice * l, s_properties *p);

void init_density(s_lattice * l);

void check_density(s_lattice *l, int time);

void redistribute(s_lattice *l, double accel, double density);

void propagate(s_lattice *l);

void bounceback(s_lattice *l);

void relaxation(s_lattice *l, double density, double omega);

double calc_velocity(s_lattice *l, int time);

void write_results(string file, s_lattice *l, double density);

void comp_rey(s_lattice *l, s_properties *p, int time, double execution); 

double crono();
