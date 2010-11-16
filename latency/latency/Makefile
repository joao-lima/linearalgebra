
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
PROGS := latency latency_async latency_async2

all: $(PROGS)

latency: latency.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

latency_async: latency_async.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

latency_async2: latency_async2.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
