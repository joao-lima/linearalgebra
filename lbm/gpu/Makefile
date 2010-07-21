
CC := nvcc
CFLAGS :=
PROGS := lb

lb: lb.cu main.cu
	nvcc  --compiler-options -fpermissive -o $@ $^ # sem o fpermissive nao funciona no gcc4.4
clean:
	rm -f $(PROGS) *.o
