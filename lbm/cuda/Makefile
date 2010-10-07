
CC := nvcc
PROGS := lb
LDFLAGS :=
NVCCFLAGS := -O2 -D_DEBUG -g
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

all: $(PROGS)
lb: lb.cu main.cu 
	$(CC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(PROGS) *.o
