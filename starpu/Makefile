
CUDA := $(CUDA_HOME)

CXX := gcc
FC          = gfortran
LOADER      = $(FC)
SCRATCH	:= $(SCRATCH)

MAGMA_CFLAGS := -I${SCRATCH}/install/magma_1.1.0/include
MAGMA_LDFLAGS := -L${SCRATCH}/install/magma_1.1.0/lib -lmagma -lmagmablas -lmagma

CXXFLAGS = -O2 -Wall -lm -pthread -lrt -DSTARPU_USE_CUDA \
	-I. \
       $(CUDA_CFLAGS) \
       $(CBLAS_CFLAGS) \
       $(LAPACKE_CFLAGS) \
       $(MAGMA_CFLAGS) \
       $(STARPU_CFLAGS)

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	  $(MAGMA_LDFLAGS) \
	  $(CUDA_LDFLAGS) \
	  -lcublas \
	  $(LAPACKE_LDFLAGS) \
	  $(LAPACK_LDFLAGS) \
	  $(CBLAS_LDFLAGS) \
	  $(STARPU_LDFLAGS)

PROGS = \
	dgemm_matprod \
	dpotrf_matcholesky 

OBJS := 

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dpotrf_matcholesky: $(OBJS) \
		cholesky/cholesky_models.o \
		cholesky/cholesky_kernels.o \
		cholesky/dpotrf_matcholesky.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

dgetrf_matlu: $(OBJS) \
		lu/lu_example_double.o \
		lu/dlu.o \
		lu/dlu_pivot.o \
		lu/dlu_kernels.o
	$(LOADER) $^ -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o lu/*.o  cholesky/*.o gemm/*.o
