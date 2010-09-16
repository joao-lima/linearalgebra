CXX := nvcc
CC := nvcc
CXXFLAGS := -O2 -D_REENTRANT -g \
	-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
LDFLAGS := $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
PROGS := origin pinned mapped pinned_wc mapped_wc

all: $(PROGS)

origin: origin.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pinned: pinned.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pinned_wc: pinned_wc.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

mapped: mapped.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

mapped_wc: mapped_wc.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
