
CUDA := /usr/local/install/cuda4.0/cuda

FC          = gfortran
LOADER      = $(FC)

CXXFLAGS = -O2 -Wall \
	 -I$(HOME)/install/magma_1.0.0/include \
	-I$(CUDA)/include \
	-I$(HOME)/install/atlas3.9.47/include \
	-I$(HOME)/install/lapacke-3.3.0/include 

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	-L$(CUDA)/lib64 -lcublas -lcuda \
	-L$(HOME)/install/lapacke-3.3.0/lib -llapacke \
	-L$(HOME)/install/atlas3.9.47/lib \
	-llapack -lcblas -latlas -lf77blas -lgfortran \
	-L$(HOME)/install/magma_1.0.0/lib -lmagma -lmagmablas \
	-llapack -lcblas -latlas -lf77blas -lgfortran 

PROGS = sgemm_matprod spotrf_matcholesky sgetrf_matlu

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

sgemm_matprod: sgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

spotrf_matcholesky: spotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgetrf_matlu: sgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
