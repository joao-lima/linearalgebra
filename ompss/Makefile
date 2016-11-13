

CC          = mnvcc
#CC          = mcc
SCRATCH = $(HOME)

LDFLAGS = -lm -lgfortran \
	/home/pbrunet/idgraf/.local/lib/libopenblas.a

CFLAGS = -DCONFIG_USE_CUDA 

PROGS = dpotrf_matcholesky

all: $(PROGS)


dgemm_matprod: dgemm_matprod.c blas.c
	$(CC) $(CFLAGS) -o $@ -O3 --ompss $^ $(LDFLAGS) 

dpotrf_matcholesky: dpotrf_matcholesky.c blas.c
	$(CC) $(CFLAGS) -o $@ -O3 --ompss $^ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
