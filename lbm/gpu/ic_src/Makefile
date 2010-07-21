
CC := nvcc
CFLAGS :=
PROGS := lb

lb: lb.cu main.cu
	nvcc  --compiler-options -fpermissive -o $@ $^
clean:
	rm -f $(PROGS) *.o
