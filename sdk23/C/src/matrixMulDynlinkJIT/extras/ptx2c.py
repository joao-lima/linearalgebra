#!/usr/bin/env python
from string import *
import os, commands, getopt, sys, platform

g_Header = '''/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

////////////////////////////////////////////////////////////////////////////////
// This file is auto-generated, do not edit
////////////////////////////////////////////////////////////////////////////////
'''


def Usage():
    print "Usage: ptx2c.py in out"
    print "Description: performs embedding in.cubin or in.ptx file into out.c and out.h files as character array" + os.linesep
    sys.exit(0)


def FormatCharHex(d):
    s = hex(ord(d))
    if len(s) == 3:
        s = "0x0" + s[2]
    return s


args = sys.argv[1:]
if not(len(sys.argv[1:]) == 2):
    Usage()

out_h = args[1] + "_ptxdump.h"
out_c = args[1] + "_ptxdump.c"


h_in = open(args[0], 'r')
source_bytes = h_in.read()

h_out_c = open(out_c, 'w')
h_out_c.writelines(g_Header)
h_out_c.writelines("#include \"" + out_h + "\"\n\n")
h_out_c.writelines("unsigned char " + args[1] + "_ptxdump[" + str(len(source_bytes)) + "] = {\n")

h_out_h = open(out_h, 'w')
macro_h = "__" + args[1] + "_ptxdump_h__"
h_out_h.writelines(g_Header)
h_out_h.writelines("#ifndef " + macro_h + "\n")
h_out_h.writelines("#define " + macro_h + "\n\n")
h_out_h.writelines('#if defined __cplusplus\nextern "C" {\n#endif\n\n')
h_out_h.writelines("extern unsigned char " + args[1] + "_ptxdump[" + str(len(source_bytes)) + "];\n\n")
h_out_h.writelines("#if defined __cplusplus\n}\n#endif\n\n")
h_out_h.writelines("#endif //" + macro_h + "\n")

newlinecnt = 0
for i in range(0, len(source_bytes)-1):
    h_out_c.write(FormatCharHex(source_bytes[i]) + ", ")
    newlinecnt += 1
    if newlinecnt == 16:
        newlinecnt = 0
        h_out_c.write("\n")
h_out_c.write(FormatCharHex(source_bytes[len(source_bytes)-1]) + "\n};\n")

h_in.close()
h_out_c.close()
h_out_h.close()

print("ptx2c: CUmodule " + args[0] + " packed successfully")
