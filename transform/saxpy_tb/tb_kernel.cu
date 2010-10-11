
#include "tb.h"

template<class Op>
__device__ void do_work( volatile tb_t *t, const float *x, float *y,
		unsigned int N, Op op ) 
{
	int index= t->i;
	int n_block= t->n;
	int i;

	for( i= 0; i < n_block; i++ )
		if( (index+i) < N )
			y[index+i]= op( x[index+i], y[index+i] );
}

template<class Op>
__global__ void tb_kernel( volatile tb_t *tb, const float *x, float *y, const unsigned
		int N, Op op )
{
	__shared__ volatile tb_t *t;
	unsigned int is_done;
	
	t= &tb[blockIdx.x];
	is_done= 0;

	while( !is_done ) {
		switch( t->status ){
		case TB_READY:
		default:
		{
			break;
		}

		case TB_POSTED:
		{
			t->status= TB_RUNNING;
			break;
		}

		case TB_RUNNING:
		{
			do_work( t, x, y, N, op );
			t->status= TB_READY;
			break;
		}

		case TB_DONE:
		{
			is_done= 1;
			break;
		}

		}
	}
}

