
CUDA := /usr/local/install/cuda4.0/cuda
SCRATCH := $(HOME)

FC          = gfortran
LOADER      = $(FC)

CXXFLAGS = -O2 -Wall \
	 -I$(SCRATCH)/install/magma_1.0.0/include \
	-I$(CUDA)/include \
	-I$(SCRATCH)/install/atlas3.9.47/include \
	-I$(SCRATCH)/install/lapacke-3.3.0/include 

CXXFLAGS_FERMI = -O2 -Wall \
	 -I$(SCRATCH)/install/magma_1.0.0-fermi/include \
	-I$(CUDA)/include \
	-I$(SCRATCH)/install/atlas3.9.47/include \
	-I$(SCRATCH)/install/lapacke-3.3.0/include 

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	-L$(CUDA)/lib64 -lcublas -lcuda \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke \
	-L$(SCRATCH)/install/atlas3.9.47/lib \
	-llapack -lcblas -latlas -lf77blas -lgfortran \
	-L$(SCRATCH)/install/magma_1.0.0/lib -lmagma -lmagmablas \
	-llapack -lcblas -latlas -lf77blas -lgfortran 

LDFLAGS_FERMI = -lm -lpthread \
	-L$(CUDA)/lib64 -lcublas -lcuda \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke \
	-L$(SCRATCH)/install/atlas3.9.47/lib \
	-llapack -lcblas -latlas -lf77blas -lgfortran \
	-L$(SCRATCH)/install/magma_1.0.0-fermi/lib -lmagma -lmagmablas \
	-llapack -lcblas -latlas -lf77blas -lgfortran 

PROGS = sgemm_matprod dgemm_matprod \
	spotrf_matcholesky \
	sgetrf_matlu \
	sgemm_matprod_fermi dgemm_matprod_fermi

all: $(PROGS)

sgemm_matprod_fermi.o: sgemm_matprod_fermi.cpp
	$(CXX) $(CXXFLAGS_FERMI) -c $< -o $@
sgemm_matprod_fermi: sgemm_matprod_fermi.o
	$(LOADER) $< -o $@ $(LDFLAGS_FERMI) 

dgemm_matprod_fermi.o: dgemm_matprod_fermi.cpp
	$(CXX) $(CXXFLAGS_FERMI) -c $< -o $@
dgemm_matprod_fermi: dgemm_matprod_fermi.o
	$(LOADER) $< -o $@ $(LDFLAGS_FERMI) 

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

sgemm_matprod: sgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

spotrf_matcholesky: spotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgetrf_matlu: sgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
