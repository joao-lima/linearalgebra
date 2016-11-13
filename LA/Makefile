

CC          = gcc
FC          = gfortran
LOADER      = $(FC)
SCRATCH = $(HOME)

LDFLAGS = -lm -lpthread -lgfortran \
	  $(LAPACKE_LDFLAGS) \
	  $(LAPACK_LDFLAGS) \
	  $(CBLAS_LDFLAGS) 

LDFLAGS_PT = -lm -lpthread -lgfortran \
	  $(LAPACKE_LDFLAGS) \
	  $(LAPACK_LDFLAGS) \
	  $(CBLAS_PT_LDFLAGS) 


CFLAGS = -O3 -Wall \
	   $(CBLAS_CFLAGS) \
	   $(LAPACKE_CFLAGS)

PROGS = dgemm_matprod \
	dgemm_pt_matprod \
	dgetrf_matlu \
	dgetrf_pt_matlu \
	dpotrf_matcholesky \
	dpotrf_pt_matcholesky \
	dgeqrf_matqr \
	dgeqrf_pt_matqr

all: $(PROGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgemm_pt_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dpotrf_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dpotrf_pt_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dgetrf_matlu: dgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgetrf_pt_matlu: dgetrf_matlu.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dgeqrf_matqr: dgeqrf_matqr.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dgeqrf_pt_matqr: dgeqrf_matqr.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

clean:
	rm -f $(PROGS) *.o
