
CXX := nvcc
CC := nvcc
CXXFLAGS := -O2 -D_REENTRANT -g \
	-I$(HOME)/cuda
LDFLAGS := 
PROGS := saxpy_thrust

all: $(PROGS)

saxpy_thrust: saxpy_thrust.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
