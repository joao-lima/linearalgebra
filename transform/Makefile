
CXX:= nvcc
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
NVCCFLAGS := -O2 -D_DEBUG
NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
PROGS:= saxpy
LDFLAGS:=

all: $(PROGS)

saxpy: saxpy.cu tb.cu
	$(CXX) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(PROGS) *.o
