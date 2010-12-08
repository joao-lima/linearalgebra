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

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define BLOCK_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA N //(3 * BLOCK_SIZE) // Matrix A width
#define HA N //(5 * BLOCK_SIZE) // Matrix A height
#define WB N //(8 * BLOCK_SIZE) // Matrix B width
#define HB N //WA  // Matrix B height
#define WC N //WB  // Matrix C width 
#define HC N //HA  // Matrix C height

#endif // _MATRIXMUL_H_

