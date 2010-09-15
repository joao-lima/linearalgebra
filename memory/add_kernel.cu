#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

// Thread block size
#define BLOCK_SIZE 64

__global__ void add_one( char *out, const char *in )
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  char res;
  char x= in[index];
  res = x + 1;
  out[index] = res;
}

#endif // #ifndef _ADD_KERNEL_H_
