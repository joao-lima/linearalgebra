
CC := nvcc
CFLAGS := -O3 -D_REENTRANT -D_CUDA_DEBUG -g
LDFLAGS := -lcuda -lcudart
PROGS := matrix_mul

all: $(PROGS)

clean:
	rm -f *.o $(PROGS)
