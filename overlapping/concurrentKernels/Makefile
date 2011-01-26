
NVCC:= nvcc
CXXFLAGS:=
LDFLAGS:= -lcuda

PROGS:= asyncKernels blockKernels concurrentKernels2 concurrentKernels \
	controlKernels duplex inverseKernels2 \
	 inverseKernels sequencialKernels

all: $(PROGS)

asyncKernels: asyncKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
blockKernels: blockKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
concurrentKernels2: concurrentKernels2.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
concurrentKernels: concurrentKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
controlKernels: controlKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
duplex: duplex.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
inverseKernels2: inverseKernels2.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
inverseKernels: inverseKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)
sequencialKernels: sequencialKernels.cu
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
