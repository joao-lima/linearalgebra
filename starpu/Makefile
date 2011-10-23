
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
	sgemm_matprod dgemm_matprod \
	spotrf_matcholesky dpotrf_matcholesky \
	sgetrf_matlu dgetrf_matlu

OBJS := blas/blas.o 

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

sgemm_matprod: $(OBJS) gemm/sgemm_matprod.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dgemm_matprod: $(OBJS) gemm/dgemm_matprod.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

spotrf_matcholesky: $(OBJS) \
		cholesky/cholesky_models.o \
		cholesky/cholesky_skernels.o \
		cholesky/spotrf_matcholesky.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dpotrf_matcholesky: $(OBJS) \
		cholesky/cholesky_models.o \
		cholesky/cholesky_dkernels.o \
		cholesky/dpotrf_matcholesky.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

sgetrf_matlu: $(OBJS) \
		lu/lu_example_float.o \
		lu/slu.o \
		lu/slu_pivot.o \
		lu/slu_kernels.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dgetrf_matlu: $(OBJS) \
		lu/lu_example_double.o \
		lu/dlu.o \
		lu/dlu_pivot.o \
		lu/dlu_kernels.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o lu/*.o  blas/*.o cholesky/*.o gemm/*.o
