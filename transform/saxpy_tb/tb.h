
#ifndef _TB_H_
#define _TB_H_

#define TB_READY 0
#define TB_POSTED 1
#define TB_RUNNING 2
#define TB_DONE 3

typedef struct {
	unsigned int status;
	unsigned int n;
	unsigned int i;
} tb_t;

void tb_init( volatile tb_t *tb );

void tb_post( volatile tb_t *tb, unsigned int t, unsigned int n );

void tb_wait( volatile tb_t *tb );

void tb_finish( volatile tb_t *tb );


#endif /* _TB_H_ */
