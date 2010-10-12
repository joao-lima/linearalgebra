
#include "cuda_defs.h"

struct saxpy_gpu 
{
    const float a;

    saxpy_gpu (float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

