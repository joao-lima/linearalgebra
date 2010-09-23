/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
 
 #ifndef _MANDELBROT_KERNEL_h_
#define _MANDELBROT_KERNEL_h_

#include <vector_types.h>

extern "C" void RunMandelbrot0_sm11(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff, 
									const double xjp, const double yjp, const double scale, const uchar4 colors, const int frame, const int animationFrame, 
									const int mode, const int numSMs, const bool isJ);
extern "C" void RunMandelbrot1_sm11(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff, 
									const double xjp, const double yjp, const double scale, const uchar4 colors, const int frame, const int animationFrame, 
									const int mode, const int numSMs, const bool isJ);

extern "C" void RunMandelbrot0_sm13(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff, 
									const double xjp, const double yjp, const double scale, const uchar4 colors, const int frame, const int animationFrame, 
									const int mode, const int numSMs, const bool isJ);
extern "C" void RunMandelbrot1_sm13(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff, 
									const double xjp, const double yjp, const double scale, const uchar4 colors, const int frame, const int animationFrame, 
									const int mode, const int numSMs, const bool isJ);

extern "C" int inEmulationMode();

#endif
