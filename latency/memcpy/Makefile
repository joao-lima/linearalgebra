
CC := gcc
CFLAGS := -D_REENTRANT -Wall -O2 -I/usr/local/cuda/include
ifeq ($(DEBUG),1)
	CFLAGS += -D_DEBUG -D_CUDA_DEBUG -g 
endif
LDFLAGS := -pthread -lcuda

PROGS := memcpy_HtoD memcpy_HtoD_pinned \
	memcpy_DtoH memcpy_DtoH_pinned

all: $(PROGS)

memcpy_HtoD: memcpy_HtoD.c
memcpy_HtoD_pinned: memcpy_HtoD_pinned.c
memcpy_DtoH: memcpy_DtoH.c
memcpy_DtoH_pinned: memcpy_DtoH_pinned.c

clean:
	rm -f *.o $(PROGS)
