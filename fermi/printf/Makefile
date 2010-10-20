
CC:= nvcc
CFLAGS := -arch=sm_20
LDFLAGS :=
PROGS := fermi_printf

all: $(PROGS)

fermi_printf: fermi_printf.cu
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o $(PROGS)
