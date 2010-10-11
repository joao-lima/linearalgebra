
#include <stdio.h>
#include "tb.h"

void tb_init( volatile tb_t *tb )
{
	tb->n= tb->i= 0;
	tb->status= TB_READY;
}

void tb_post( volatile tb_t *tb, unsigned int t, unsigned int n )
{
	fprintf( stdout, "tb_post pos=%d n=%d\n", t, n ); fflush(stdout);
	tb->i= t;
	tb->n= n;
	__asm__ __volatile__ ("" ::: "memory");
	tb->status= TB_POSTED;
}

void tb_wait( volatile tb_t *tb )
{
	fprintf( stdout, "tb_wait pos=%d n=%d\n", tb->i, tb->n );
	fflush(stdout);
	while( tb->status != TB_READY ) ;
}

void tb_finish( volatile tb_t *tb )
{
	tb->status= TB_DONE;
}
