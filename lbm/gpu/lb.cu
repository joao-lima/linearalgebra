
#include <iostream>
#include <fstream>

#include "lb.h"

lb::lb() {}

void lb::read( const char *parameters, const char *obstacles )
{
	std::ifstream par, obs;
	int max, c=0;
	int i, j;

	par.open( parameters );
	obs.open( obstacles );
	if( !par.is_open() || !obs.is_open() )
		return;
	par >> max_iter;
	par >> density;
	par >> accel;
	par >> omega;
	par >> r_rey;

	obs >> nx;
	obs >> ny;
	obs >> ndim;
	obs >> max;

	std::cout << "nx=" << nx << " ny=" << ny << " ndim=" << ndim 
		<< std::endl;
	resize( nx * ny );
	while( c < max ){
		obs >> i;
		obs >> j;
		obst[pos(i-1,j-1)] = true;
		c++;
	}
	par.close();
	obs.close();
}

void lb::resize( const int n )
{
	f0.resize( n ); obst.resize( n );
	f1.resize( n ); f2.resize( n ); f3.resize( n ); f4.resize( n );
	f5.resize( n ); f6.resize( n ); f7.resize( n ); f8.resize( n );
}

void lb::init( )
{
	int x, y;
	double t_0 = density * 4.0 / 9.0;
	double t_1 = density / 9.0;
	double t_2 = density / 36.0;

	//loop over computational domain
	for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
			//zero velocity density
			f0[pos(x,y)] = t_0;
			//equilibrium densities for axis speeds
			f1[pos(x,y)] = t_1;
			f2[pos(x,y)] = t_1;
			f3[pos(x,y)] = t_1;
			f4[pos(x,y)] = t_1;
			//equilibrium densities for diagonal speeds
			f5[pos(x,y)] = t_2;
			f6[pos(x,y)] = t_2;
			f7[pos(x,y)] = t_2;
			f8[pos(x,y)] = t_2;
		}
	}
}

/* essa função pode ter uma implementação CUDA/thrust 
   eu vi uma função chamada transform_reduce, quem sabe ...
*/
float lb::velocity( int time ) 
{
	int x, y, n_free;
	double u_x, d_loc;

	x = nx/2;
	n_free = 0;
	u_x = 0;

	for( y = 0; y < ny; y++ ) {
		if ( obst[pos(x,y)] == false ){
			d_loc = d_loc + f0[pos(x,y)];
			d_loc += d_loc + f1[pos(x,y)];
			d_loc += d_loc + f2[pos(x,y)];
			d_loc += d_loc + f3[pos(x,y)];
			d_loc += d_loc + f4[pos(x,y)];
			d_loc += d_loc + f5[pos(x,y)];
			d_loc += d_loc + f6[pos(x,y)];
			d_loc += d_loc + f7[pos(x,y)];
			d_loc += d_loc + f8[pos(x,y)];
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
}

//////////////////////////////////////////
// Redistribute
//////////////////////////////////////////
//ESSE KERNEL TEM DE SER CHAMADO COM APENAS UMA DIMENSAO
__global__ void redistribute(float * f1,float * f3,float * f5,float * f6,float * f7,float * f8,bool* obst, 
              double accel, double density, int nx, int ny) {
    //nx e ny sao as dimensoes
    //local variables
    int y;
    double t_1 = density * accel / 9.0;
    double t_2 = density * accel / 36.0;

    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row >= ny) return;
    //for (y = 0; y < l->ly; y++) {
    //check to avoid negative densities
    //check false | true
    if ((obst[row * nx] == false) && ((f3[row * nx] - t_1) > 0) && 
                 ((f6[row * nx] - t_2) > 0) && (f7[row * nx] - t_2 > 0)) {
      //increase east
      f1[row * nx] += t_1;
      //l->node[0][y][1] += t_1;
      //decrease west
      f3[row * nx] -= t_1;
      //l->node[0][y][3] -= t_1;
      //increase north-east
      f5[row * nx] += t_2;
      //l->node[0][y][5] += t_2;
      //decrease north-west
      f6[row * nx] -= t_2;
      //l->node[0][y][6] -= t_2;
      //decrease south-west
      f7[row * nx] -= t_2;
      //l->node[0][y][7] -= t_2;
      //increase south-east
      f8[row * nx] += t_2;
      //l->node[0][y][8] += t_2;
    }
  //}
}

void lb::redistribute( void )
{
	/* here a kernel call */
// tem de chamar esse kernel com uma dimensao apenas

}

void lb::propagate( void )
{
	/* here a kernel call */
}

//////////////////////////////////////////
// Bounceback
//////////////////////////////////////////

__global__ void bounceback(float * f1,float * f2,float * f3,float * f4,float * f5,float * f6,float * f7,float * f8,
            float * tf1,float * tf2,float * tf3,float * tf4,float * tf5,float * tf6,float * tf7,float * tf8,bool* obst,
            int nx, int ny) {
  //local variables
  //TODO ver o acesso a memoria. nao fica totalmente desalinhado usando 8 vetores nao?
  //-- indexes
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

      //como sei qual as dimensoes totais das matrizes
      if ((row > dimx) or (col > dimy)) return;//verifica quais threads devem executar
      if (obst[row * nx + col]){
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
  }
}




void lb::bounceback( void )
{
	/* here a kernel call */
}

void lb::relaxation( void )
{
	/* here a kernel call */
}
