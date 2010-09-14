CXX := nvcc
CC := nvcc
CXXFLAGS := -O2 -D_REENTRANT -D_DEBUG -g \
	-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
LDFLAGS := $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
PROGS := matrixMul matrixMul_pinned
SRC := matrixMul_gold.cpp 
SRC1 := matrixMul.cu $(SRC) 
SRC2 := matrixMul_pinned.cu $(SRC) 

all: $(PROGS)

matrixMul: $(SRC1) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC1) $(LDFLAGS)

matrixMul_pinned: $(SRC2) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC2) $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
