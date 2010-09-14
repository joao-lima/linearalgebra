CXX := nvcc
CC := nvcc
CXXFLAGS := -O3 -D_REENTRANT  \
	-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
LDFLAGS := $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
PROGS := matrixMul matrixMul_pinned matrixMul_mapped
SRC := matrixMul_gold.cpp 
SRC1 := matrixMul.cu $(SRC) 
SRC2 := matrixMul_pinned.cu $(SRC) 
SRC3 := matrixMul_mapped.cu $(SRC) 

all: $(PROGS)

matrixMul: $(SRC1) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC1) $(LDFLAGS)

matrixMul_pinned: $(SRC2) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC2) $(LDFLAGS)

matrixMul_mapped: $(SRC3) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC3) $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
