
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
CXXFLAGS := -O2 -D_REENTRANT
ifeq ($(DEBUG),1)
	CXXFLAGS += -D_DEBUG -g $(NVCCFLAGS)
endif

LDFLAGS :=
PROGS := latency_async_nomem latency_async_mem latency_async_mem_misaligned 

all: $(PROGS)

latency_async_nomem: latency_async_nomem.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

latency_async_mem: latency_async_mem.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

latency_async_mem_misaligned: latency_async_mem_misaligned.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
