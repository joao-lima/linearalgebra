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

#ifndef _PTXJIT_H_
#define _PTXJIT_H_

char myPtx[] = "\n\
	.version 1.4\n\
	.target sm_10, map_f64_to_f32\n\
	.reg .u32 %ra<17>;\n\
	.reg .u64 %rda<17>;\n\
	.reg .f32 %fa<17>;\n\
	.reg .f64 %fda<17>;\n\
	.reg .u32 %rv<5>;\n\
	.reg .u64 %rdv<5>;\n\
	.reg .f32 %fv<5>;\n\
	.reg .f64 %fdv<5>;\n\
    .entry _Z8myKernelPi (\n\
		.param .u32 __cudaparm__Z8myKernelPi_data)\n\
	{\n\
	.reg .u16 %rh<4>;\n\
	.reg .u32 %r<8>;\n\
	.loc	14	1	0\n\
$LBB1__Z8myKernelPi:\n\
	.loc	14	4	0\n\
	mov.u16 	%rh1, %ctaid.x;\n\
	mov.u16 	%rh2, %ntid.x;\n\
	mul.wide.u16 	%r1, %rh1, %rh2;\n\
	cvt.u32.u16 	%r2, %tid.x;\n\
	add.u32 	%r3, %r2, %r1;\n\
	ld.param.u32 	%r4, [__cudaparm__Z8myKernelPi_data];\n\
	mul.lo.u32 	%r5, %r3, 4;\n\
	add.u32 	%r6, %r4, %r5;\n\
	st.global.s32 	[%r6+0], %r3;\n\
	.loc	14	5	0\n\
	exit;\n\
$LDWend__Z8myKernelPi:\n\
	}\n\
";

#endif
