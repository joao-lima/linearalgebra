/* The densities are numbered as follows:
 *               6   2   5
 *                 \ | /
 *               3 - 0 - 1
 *                 / | \
 *               7   4   8
 *
 * The lattice nodes are numbered as follows:
 *      ^
 *      |
 *      y
 *           :    :    :
 *
 *      3    *    *    *  ..
 *
 *           *    *    *  ..
 *
 *      1    *    *    *  ..
 *                            x ->
 *           1    2    3 
 *      
*/


///////////////////////////////////////////////
//libraries
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define FALSE 0
#define TRUE 1

#define SIZE_OF_LATTICE 512

#define THREADS 16
#define BLOCKS 8

//typedef enum {false, true} bool; // <- não funciona com CUDA!
typedef char *string;


///////////////////////////////////////////////
//structs

//struct contends macroscopic information
typedef struct{
	int t_max; //Maximum number of iterations
	double density; //Fluid density per link
	double accel; //Accelleration
	double omega; //Relaxation parameter
	double r_rey; //Linear dimension for Reynolds number
	double vel, visc, rey; // Reynolds number information
} s_properties;

//lattice structure
typedef struct {
	int lx; //nodes number in axis x
	int ly; //nodes number in axis y
	int n; //lattice dimension elements
	bool obst[512][512]; //**obst; //Obstacle Array lx * ly
	double node[512][512][9]; //***node; //n-speed lattice  n * lx * ly
	double temp[512][512][9]; //***temp; //temporarely storage of fluid densities
	double n_sum; //*n_sum; // check_density variable
} s_lattice;

///////////////////////////////////////////////
//GPU Functions

__global__ void GPUmain_loop(s_lattice *l, s_properties *p, int time);

__global__ void GPUinit_density(s_lattice * l, double density);
__global__ void GPUcheck_density(s_lattice *l, double *vel);
__global__ void GPUcomp_rey(s_lattice *l, s_properties *p, int time);
__global__ void GPUcalc_velocity(s_lattice *l, int time, double *vel);

__global__ void GPUredistribute(s_lattice *l, double accel, double density);
__global__ void GPUpropagate(s_lattice *l);
__global__ void GPUbounceback(s_lattice *l);
__global__ void GPUrelaxation(s_lattice *l, double density, double omega);

///////////////////////////////////////////////
//Functions

s_properties* read_parametrs(string file);

void alloc_lattice(s_lattice *l);

s_lattice* read_obstacles(string file);

void init_density(s_lattice * l, double density);

void check_density(s_lattice *l, int time);

void redistribute(s_lattice *l, double accel, double density);

void propagate(s_lattice *l);

void bounceback(s_lattice *l);

void relaxation(s_lattice *l, double density, double omega);

double calc_velocity(s_lattice *l, int time);

void write_results(string file, s_lattice *l, s_properties *p, int time, double execution, int griddim, int blockdim);

void comp_rey(s_lattice *l, s_properties *p, int time, double execution); 

double crono();
