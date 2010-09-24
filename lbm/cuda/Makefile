
CC := nvcc
CFLAGS := 
PROGS := lb
LDFLAGS :=

all: $(PROGS)
lb: lb.cu main.cu
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(PROGS) *.o
