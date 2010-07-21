
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
	float t_0 = density * 4.0 / 9.0;
	float t_1 = density / 9.0;
	float t_2 = density / 36.0;

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
	float u_x, d_loc;

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

void lb::redistribute( void )
{
	/* here a kernel call */
}

void lb::propagate( void )
{
	/* here a kernel call */
}

//////////////////////////////////////////
// Bounceback
//////////////////////////////////////////

__global__ void bounceback(
	float * f1, float * f2, float * f3, float * f4, float * f5,
	float * f6, float * f7, float * f8,
	float * tf1, float * tf2, float * tf3, float * tf4, float * tf5,
	float * tf6, float * tf7, float * tf8, bool* obst, 
	int nx, int ny
	) 
{
  //local variables
  //TODO ver o acesso a memoria. nao fica totalmente desalinhado usando 8 vetores nao?
  //TODO SIZE, dimx, dimy precisam ser definido. se ja estiver preciso saber qual eh
  int SIZE = -1, dimx = -1, dimy = -1;
  //-- indexes
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

      //como sei qual as dimensoes totais das matrizes
      if ((row > dimx) or (col > dimy)) return;//verifica quais threads devem executar
      if (obst[row * SIZE + col]){
        //east
        f1[row * SIZE + col] = tf3[row * SIZE + col];
        //north
        f2[row * SIZE + col] = tf4[row * SIZE + col];
        //west
        f3[row * SIZE + col] = tf1[row * SIZE + col];
        //south
        f4[row * SIZE + col] = tf2[row * SIZE + col];
        //north-east
        f5[row * SIZE + col] = tf7[row * SIZE + col];
        //north-west
        f6[row * SIZE + col] = tf8[row * SIZE + col];
        //south-west
        f7[row * SIZE + col] = tf5[row * SIZE + col];
        //south-east
        f8[row * SIZE + col] = tf6[row * SIZE + col];
      }
    }
  }
}




void lb::bounceback( void )
{
	/* here a kernel call */
}

__global__ void relaxation( 
	float *f0, float * f1, float * f2, float * f3, float * f4, float * f5,
	float * f6, float * f7, float * f8,
	float *tf0, float * tf1, float * tf2, float * tf3, float * tf4, 
	float *tf5, float * tf6, float * tf7, float * tf8, bool* obst,
	int nx, int ny, float omega
	)
{
	//local variables
	float c_squ = 1.0 / 3.0;
	float t_0 = 4.0 / 9.0;
	float t_1 = 1.0 / 9.0;
	float t_2 = 1.0 / 36.0;
	float u_x, u_y;
	float u_n[9], n_equ[9], u_squ, d_loc;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if( (y > ny) || (x > nx) ) return;
	if ( obst[pos(x,y)] == false ) {
		d_loc = d_loc + tf0[pos(x,y)];
		d_loc += d_loc + tf1[pos(x,y)];
		d_loc += d_loc + tf2[pos(x,y)];
		d_loc += d_loc + tf3[pos(x,y)];
		d_loc += d_loc + tf4[pos(x,y)];
		d_loc += d_loc + tf5[pos(x,y)];
		d_loc += d_loc + tf6[pos(x,y)];
		d_loc += d_loc + tf7[pos(x,y)];
		d_loc += d_loc + tf8[pos(x,y)];

		//x-, and y- velocity components
		u_x = (tf1[pos(x,y)] + tf5[pos(x,y)] + tf8[pos(x,y)] - (tf3[pos(x,y)] + tf6[pos(x,y)] + tf7[pos(x,y)])) / d_loc;
		//u_x = (l->temp[x][y][1] + l->temp[x][y][5] + l->temp[x][y][8] - (l->temp[x][y][3] + l->temp[x][y][6] + l->temp[x][y][7])) / d_loc;

		u_y = (tf2[pos(x,y)] + tf5[pos(x,y)] + tf6[pos(x,y)] - (tf4[pos(x,y)] + tf7[pos(x,y)] + tf8[pos(x,y)])) / d_loc;
		//u_y = (l->temp[x][y][2] + l->temp[x][y][5] + l->temp[x][y][6] - (l->temp[x][y][4] + l->temp[x][y][7] + l->temp[x][y][8])) / d_loc;

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
		f0[pos(x,y)] = tf0[pos(x,y)] + omega *
		       	(n_equ[0] - tf0[pos(x,y)]);
		f1[pos(x,y)] = tf1[pos(x,y)] + omega *
		       	(n_equ[1] - tf1[pos(x,y)]);
		f2[pos(x,y)] = tf2[pos(x,y)] + omega *
		       	(n_equ[2] - tf2[pos(x,y)]);
		f3[pos(x,y)] = tf3[pos(x,y)] + omega *
		       	(n_equ[3] - tf3[pos(x,y)]);
		f4[pos(x,y)] = tf4[pos(x,y)] + omega *
		       	(n_equ[4] - tf4[pos(x,y)]);
		f5[pos(x,y)] = tf5[pos(x,y)] + omega *
		       	(n_equ[5] - tf5[pos(x,y)]);
		f6[pos(x,y)] = tf6[pos(x,y)] + omega *
		       	(n_equ[6] - tf6[pos(x,y)]);
		f7[pos(x,y)] = tf7[pos(x,y)] + omega *
		       	(n_equ[7] - tf7[pos(x,y)]);
		//for (i = 0; i < l->n; i++) {
		//	l->node[x][y][i] = l->temp[x][y][i] + omega * (n_equ[i] - l->temp[x][y][i]);
		//}	
	}
}

void lb::relaxation( void )
{
	/* here a kernel call */
}
