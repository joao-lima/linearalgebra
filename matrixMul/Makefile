CXX := nvcc
CC := nvcc
CXXFLAGS := -O2 -D_REENTRANT -D_DEBUG -g \
	-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
LDFLAGS := $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
PROGS := matrixMul
SRC := matrixMul.cu matrixMul_gold.cpp 

all: $(PROGS)

matrixMul: $(SRC) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
