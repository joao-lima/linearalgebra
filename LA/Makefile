

CC          = gcc
FC          = gfortran
LOADER      = $(FC)
SCRATCH = $(HOME)

LDFLAGS = -lm \
	-L$(SCRATCH)/install/atlas3.9.47/lib -llapack -lcblas -latlas -lf77blas \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke  -llapack

LDFLAGS_PT = -lm -pthread \
	-L$(SCRATCH)/install/atlas3.9.47/lib -llapack -lptcblas -latlas -lptf77blas \
	-L$(SCRATCH)/install/lapacke-3.3.0/lib -llapacke  -llapack

CFLAGS = -Wall -O2 -DADD_ -I$(SCRATCH)/install/atlas3.9.47/include \
	-I$(SCRATCH)/install/lapacke-3.3.0/include 

PROGS = sgemm_matprod  dgemm_matprod \
	sgemm_pt_matprod  dgemm_pt_matprod \
	sgetrf_matlu dgetrf_matlu \
	sgetrf_pt_matlu dgetrf_pt_matlu \
	spotrf_matcholesky dpotrf_matcholesky \
	spotrf_pt_matcholesky dpotrf_pt_matcholesky

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

sgetrf_matlu: sgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgetrf_matlu: dgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

sgetrf_pt_matlu: sgetrf_pt_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dgetrf_pt_matlu: dgetrf_pt_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

spotrf_pt_matcholesky: spotrf_pt_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dpotrf_pt_matcholesky: dpotrf_pt_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

clean:
	rm -f $(PROGS) *.o
