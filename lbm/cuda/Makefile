
CC := nvcc
CFLAGS := -D_DEBUG -O2 -g
PROGS := lb
LDFLAGS :=

all: $(PROGS)
lb: lb.cu main.cu
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(PROGS) *.o
