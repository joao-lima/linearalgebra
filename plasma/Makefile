
include make.inc

CUDA := /usr/local/install/cuda4.0/cuda

CFLAGS +=  -I$(SCRATCH)/install/plasma-2.4.1/include

#LDFLAGS += $(LIBBLAS)    $(LIBCBLAS)   $(LIBLAPACK) $(LIBCLAPACK)
LDFLAGS += -L$(SCRATCH)/install/plasma-2.4.1/lib \
-lplasma  -lcoreblas -lplasma -lquark \
-L$(SCRATCH)/install/plasma-2.4.1/lib -lcblas \
-L$(SCRATCH)/install/plasma-2.4.1/lib -llapacke \
-L$(SCRATCH)/install/plasma-2.4.1/lib -ltmg -llapack \
-L$(SCRATCH)/install/plasma-2.4.1/lib -lrefblas  -lpthread -lm

PROGS = spotrf_matcholesky sgetrf_matlu sgemm_matprod \
	dgemm_matprod

all: $(PROGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

spotrf_matcholesky: spotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgetrf_matlu: sgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgemm_matprod: sgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
