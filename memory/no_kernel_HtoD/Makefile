
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
PROGS := origin pinned pinned_wc pinned_async

all: $(PROGS)

origin: origin.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pinned: pinned.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pinned_wc: pinned_wc.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pinned_async: pinned_async.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
