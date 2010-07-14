
CC := nvcc
CFLAGS :=
PROGS := lb

lb: lb.cu main.cu
	nvcc -o $@ $^
clean:
	rm -f $(PROGS) *.o
