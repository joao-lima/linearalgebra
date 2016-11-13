

CC          = icc
LOADER      = icc
#SCRATCH = $(SCRATCH)

LDFLAGS = -lm -mkl=sequential 

LDFLAGS_PT = -lm -mkl=parallel 

CFLAGS = -O3 -Wall \
	 -I/usr/include/x86_64-linux-gnu  

PROGS = dgemm_matprod  \
	dgemm_pt_matprod  \
	dpotrf_matcholesky \
	dpotrf_pt_matcholesky

all: $(PROGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

dgemm_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dpotrf_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS) 

dpotrf_pt_matcholesky: dpotrf_matcholesky.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

dgemm_pt_matprod: dgemm_matprod.o
	$(LOADER) $< -o $@ $(LDFLAGS_PT) 

clean:
	rm -f $(PROGS) *.o
