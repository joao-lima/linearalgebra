
CXX := nvcc
CC := nvcc
# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function
NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
CXXFLAGS := -O2 -D_REENTRANT  $(NVCCFLAGS) -g -D_DEBUG
LDFLAGS := -lcublas
PROGS := simpleCUBLAS cublas_sgemm cublas_sgemm_pinned \
	cublas_sgemm_async1 cublas_sgemm_async2

all: $(PROGS)

simpleCUBLAS: simpleCUBLAS.c
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

cublas_sgemm: cublas_sgemm.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

cublas_sgemm_pinned: cublas_sgemm_pinned.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

cublas_sgemm_async1: cublas_sgemm_async1.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

cublas_sgemm_async2: cublas_sgemm_async2.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
