CXX := nvcc
CC := nvcc
CXXFLAGS := -O3 -D_REENTRANT  \
	-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
LDFLAGS := $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
PROGS := matrixMul matrixMul_pinned matrixMul_mapped \
	matrixMul_pinned_wc matrixMul_async1 matrixMul_async2
SRC := matrixMul_gold.cpp 
SRC1 := matrixMul.cu $(SRC) 
SRC2 := matrixMul_pinned.cu $(SRC) 
SRC3 := matrixMul_mapped.cu $(SRC) 
SRC4 := matrixMul_pinned_wc.cu $(SRC) 
SRC5 := matrixMul_async1.cu $(SRC) 
SRC6 := matrixMul_async2.cu $(SRC) 

all: $(PROGS)

matrixMul: $(SRC1) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC1) $(LDFLAGS)

matrixMul_pinned: $(SRC2) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC2) $(LDFLAGS)

matrixMul_mapped: $(SRC3) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC3) $(LDFLAGS)

matrixMul_pinned_wc: $(SRC4) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC4) $(LDFLAGS)

matrixMul_async1: $(SRC5) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC5) $(LDFLAGS)

matrixMul_async2: $(SRC6) matrixMul.h matrixMul_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $(SRC6) $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
