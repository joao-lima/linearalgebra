CXX := nvcc
CC := nvcc
CXXFLAGS := -O3 -D_REENTRANT 
LDFLAGS := -lcuda -lcublas
PROGS := simpleCUBLAS_async simpleCUBLAS

all: $(PROGS)

simpleCUBLAS:  simpleCUBLAS.cu
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

simpleCUBLAS_async:  simpleCUBLAS_async.cu
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
