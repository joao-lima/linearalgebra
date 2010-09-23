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

#ifndef __SOBELFILTER_KERNELS_H_
#define __SOBELFILTER_KERNELS_H_

#define TILEX   36 // Sobel tile width
#define TILEY   28 // Sobel tile height
#define FTILEX  38 // Filter tile width
#define FTILEY  30 // Filter tile height
#define PTILEX  48 // TILEX + 2*RADIUS+1
#define PTILEY  40 // TILEY + 2*RADIUS+1 
#define TIDSX   48 // Threads in X
#define TIDSY   8  // Threads in Y
#define RADIUS  5  // Filter radius
#define RPIXELS  5  // Pixels per thread
#define WPIXELS  4  // Pixels per thread

typedef unsigned char Pixel;

// global determines which filter to invoke
enum SobelDisplayMode {
	SOBELDISPLAY_IMAGE = 0,
	SOBELDISPLAY_SOBELTEX,
	SOBELDISPLAY_SOBELSHARED
};

static char *filterMode[] = { 
	"Original Filter", 
	"Sobel Shared", 
	"Sobel Shared+Texture", 
	NULL 
};

extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);

#endif

