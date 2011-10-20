
CUDA := /usr/local/install/cuda4.0/cuda

CXX := gcc
FC          = gfortran
LOADER      = $(FC)
SCRATCH	:= $(HOME)

CXXFLAGS = -O2 -Wall -lm -pthread -lrt \
	-I. \
	-I$(CUDA)/include \
	-I$(SCRATCH)/install/atlas3.9.47/include \
	-I$(SCRATCH)/install/lapacke-3.3.0/include  \
	-I$(SCRATCH)/install/starpu-0.9.2/include

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	-L$(CUDA)/lib64 -lcublas -lcuda -lOpenCL \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke \
	-L$(SCRATCH)/install/atlas3.9.47/lib \
	-L$(SCRATCH)/install/starpu-0.9.2/lib \
	-llapack -lcblas -latlas -lf77blas -lgfortran -lstarpu

PROGS = \
	sgemm_matprod dgemm_matprod

OBJS := blas.o 

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

sgemm_matprod: $(OBJS) sgemm_matprod.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dgemm_matprod: $(OBJS) dgemm_matprod.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
