
CUDA := /usr/local/install/cuda4.0/cuda

FC          = gfortran
LOADER      = $(FC)
SCRATCH	:= $(HOME)

CXXFLAGS = -O2 -Wall \
	-I$(CUDA)/include \
	-I$(SCRATCH)/install/atlas3.9.47/include \
	-I$(SCRATCH)/install/lapacke-3.3.0/include 

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	-L$(CUDA)/lib64 -lcublas -lcuda \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke \
	-L$(SCRATCH)/install/atlas3.9.47/lib \
	-llapack -lcblas -latlas -lf77blas -lgfortran 

PROGS = \
	sgemm_matprod dgemm_matprod

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

sgemm_matprod: sgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
