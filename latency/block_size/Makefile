
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
PROGS := timed_kernel_clock_nowork timed_kernel_clock_work

all: $(PROGS)

timed_kernel_clock_nowork: timed_kernel_clock_nowork.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

timed_kernel_clock_work: timed_kernel_clock_work.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
