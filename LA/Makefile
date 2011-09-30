

CC          = gcc
FC          = gfortran
LOADER      = $(FC)

LDFLAGS = -lm \
	-L$(HOME)/install/atlas3.9.47/lib -llapack -lcblas -latlas -lf77blas \
	-L$(HOME)/install/lapacke-3.3.0/lib -llapacke  -llapack

LDFLAGS_PT = -lm -pthread \
	-L$(HOME)/install/atlas3.9.47/lib -llapack -lptcblas -latlas -lptf77blas \
	-L$(HOME)/install/lapacke-3.3.0/lib -llapacke  -llapack

CFLAGS = -Wall -O2 -DADD_ -I$(HOME)/install/atlas3.9.47/include \
	-I$(HOME)/install/lapacke-3.3.0/include 

PROGS = sgemm_matprod  dgemm_matprod \
	sgemm_pt_matprod  dgemm_pt_matprod \
	spotrf_matcholesky dpotrf_matcholesky

all: $(PROGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

sgemm_matprod: sgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgemm_pt_matprod: sgemm_pt_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dgemm_pt_matprod: dgemm_pt_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

spotrf_matcholesky: spotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dpotrf_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(PROGS) *.o
