
CUDA := $(CUDA_HOME)
SCRATCH := /scratch/jvlima

FC          = g++
LOADER      = $(FC)

CXXFLAGS = -O3 -Wall \
	   $(CUDA_CFLAGS) \
	   $(CBLAS_CFLAGS) \
	   $(LAPACKE_CFLAGS) \
	   $(MAGMA_CFLAGS)

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread -lgfortran \
	  $(MAGMA_LDFLAGS) \
	  $(CUDA_LDFLAGS) \
	  -lcublas \
	  $(LAPACKE_LDFLAGS) \
	  $(LAPACK_LDFLAGS) \
	  $(CBLAS_LDFLAGS) 

PROGS = \
	dgemm_matprod \
	dgemm_matprod_nocopy \
	dgetrf_matlu \
	dgeqrf_matqr \
	dpotrf_matcholesky

all: $(PROGS)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod_nocopy: dgemm_matprod_nocopy.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dpotrf_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgetrf_matlu: dgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgeqrf_matqr: dgeqrf_matqr.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
