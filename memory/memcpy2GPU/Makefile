
CC := gcc
CFLAGS := -D_REENTRANT -Wall -O2 -I/usr/local/cuda/include
ifeq ($(DEBUG),1)
	CFLAGS += -D_DEBUG -D_CUDA_DEBUG -g 
endif
LDFLAGS := -pthread -lcuda

PROGS := memcpy_HtoD

all: $(PROGS)

memcpy_HtoD: memcpy_HtoD.c

clean:
	rm -f *.o $(PROGS)
