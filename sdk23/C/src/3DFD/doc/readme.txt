This sample illustrates a 3D stencil computation over a uniform grid, a
computation common in finite difference codes.  The kernel advances 2D
threadblocks along the slowest-varying dimension of the 3D data set.
Data is kept in registers and shared memory for each computation, thus
effectively streaming the input.  Data ends up being read twice, due to 
the halos (16x16 output region for each threadblock, 4 halo regions, each
16x4).  For more details please refer to:

 *  P. Micikevicius, 3D finite difference computation on GPUs using CUDA. In 
    Proceedings of 2nd Workshop on General Purpose Processing on Graphics 
    Processing Units (Washington, D.C., March 08 - 08, 2009). GPGPU-2, 
    vol. 383. ACM, New York, NY, 79-84.
  
 *  CUDA Optimization slides, Supercomputing 08 CUDA totorial
    http://gpgpu.org/static/sc2008/M02-04_Optimization.pdf