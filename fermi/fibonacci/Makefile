
CC:= nvcc
CFLAGS := -arch=sm_20
LDFLAGS :=
PROGS := fermi_fibo

all: $(PROGS)

fermi_fibo: fermi_fibo.cu
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o $(PROGS)
