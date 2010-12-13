CXX := nvcc
CC := nvcc
CXXFLAGS := -O3 -D_REENTRANT 
LDFLAGS := 
PROGS := matrixMul matrixMul_async matrixMul_memcpy
SRC := matrixMul_gold.cpp 
SRC1 := matrixMul_async.cu $(SRC) 
SRC2 := matrixMul.cu $(SRC) 
SRC3 := matrixMul_memcpy.cu $(SRC)

all: $(PROGS)

matrixMul_async: $(SRC1) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC1) $(LDFLAGS)

matrixMul: $(SRC2) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC2) $(LDFLAGS)

matrixMul_memcpy: $(SRC3) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC3) $(LDFLAGS)
	

clean:
	rm -f $(PROGS) *.o
