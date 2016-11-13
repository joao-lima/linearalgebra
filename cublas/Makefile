
CUDA := $(CUDA_HOME)

FC          = gfortran
LOADER      = $(FC)
SCRATCH	:= /scratch/jvlima
CC := gcc

CXXFLAGS = -O3 -Wall \
	   $(CUDA_CFLAGS) \
	   $(CBLAS_CFLAGS) \
	   $(LAPACKE_CFLAGS)

CFLAGS := $(CXXFLAGS)

LDFLAGS = -lm -lpthread \
	  $(CUDA_LDFLAGS) \
	  -lcublas \
	  $(LAPACKE_LDFLAGS) \
	  $(LAPACK_LDFLAGS) \
	  $(CBLAS_LDFLAGS) 

OBJS := pool/kaapi_cuda_pool.o
PROGS = \
	dgemm_matprod \
	dgemm_matprod_nocopy \
	dgemm_matprod_register \
	dgemm_matprod_anod2h \
	dgemm_matprod_async 

all: $(PROGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod_nocopy: dgemm_matprod_nocopy.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod_async: dgemm_matprod_async.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod_register: dgemm_matprod_register.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod_anod2h: dgemm_matprod_anod2h.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o */*.o
