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

/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates 
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load 
* the CUDA ptx (*.ptx) kernel just prior to kernel launch.
* 
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, CUDA
#include <cutil_inline.h>

// includes, project
#include <cutil_inline.h>

char *image_filename = "lena_bw.pgm";
char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

static CUresult initCUDA(int argc, char**argv, CUfunction*);

const char *sSDKsample = "simpleTextureDrv (Driver API)";

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

void
showHelp()
{
  printf("\n> Command line options\n", sSDKsample);
  printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    printf("[ %s ]\n", sSDKsample);

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "help") ) {
        showHelp();
        return 0;
    }

    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // initialize CUDA
    CUfunction transform = NULL;
    cutilDrvSafeCall(initCUDA(argc, argv, &transform));

    // load image from disk
    float* h_data = NULL;
    unsigned int width, height;
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == 0)
        exit(EXIT_FAILURE);
    cutilCheckError( cutLoadPGMf(image_path, &h_data, &width, &height));

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

    // load reference image from image (output)
    float *h_data_ref = (float*) malloc(size);
    char* ref_path = cutFindFilePath(ref_filename, argv[0]);
    if (ref_path == 0) {
        printf("Unable to find reference file %s\n", ref_filename);
        exit(EXIT_FAILURE);
    }
    cutilCheckError( cutLoadPGMf(ref_path, &h_data_ref, &width, &height));

    // allocate device memory for result
    CUdeviceptr d_data = (CUdeviceptr)NULL;
    cutilDrvSafeCall( cuMemAlloc( &d_data, size));

    // allocate array and copy image data
    CUarray cu_array;
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    cutilDrvSafeCall( cuArrayCreate( &cu_array, &desc ));
	CUDA_MEMCPY2D copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copyParam.dstArray = cu_array;
	copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParam.srcHost = h_data;
	copyParam.srcPitch = width * sizeof(float);
	copyParam.WidthInBytes = copyParam.srcPitch;
	copyParam.Height = height;
    cutilDrvSafeCall(cuMemcpy2D(&copyParam));

    // set texture parameters
    CUtexref cu_texref;
    cutilDrvSafeCall(cuModuleGetTexRef(&cu_texref, cuModule, "tex"));
    cutilDrvSafeCall(cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT));
    cutilDrvSafeCall(cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_WRAP));
    cutilDrvSafeCall(cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_WRAP));
    cutilDrvSafeCall(cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR));
    cutilDrvSafeCall(cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES));
    cutilDrvSafeCall(cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_FLOAT, 1));

    // set kernel parameters
    int offset = 0;
    void* ptr = (void*)(size_t)d_data;
    offset = (offset + __alignof(ptr) - 1) & ~(__alignof(ptr) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(transform, offset, &ptr, sizeof(ptr)));
    offset += sizeof(ptr);

    offset = (offset + __alignof(width) - 1) & ~(__alignof(width) - 1);
    cutilDrvSafeCall(cuParamSeti(transform, offset, width));
    offset += sizeof(width);

    offset = (offset + __alignof(height) - 1) & ~(__alignof(height) - 1);
    cutilDrvSafeCall(cuParamSeti(transform, offset, height));
    offset += sizeof(height);

    offset = (offset + __alignof(angle) - 1) & ~(__alignof(angle) - 1);
    cutilDrvSafeCall(cuParamSetf(transform, offset, angle));
    offset += sizeof(angle);

    cutilDrvSafeCall(cuParamSetSize(transform, offset));
    cutilDrvSafeCall(cuParamSetTexRef(transform, CU_PARAM_TR_DEFAULT, cu_texref));

    // set execution configuration
	int block_size = 8;
    cutilDrvSafeCall(cuFuncSetBlockShape( transform, block_size, block_size, 1 ));

    // warmup
    cutilDrvSafeCall(cuLaunchGrid( transform, width / block_size, height / block_size ));

    cutilDrvSafeCall( cuCtxSynchronize() );
    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // execute the kernel
    cutilDrvSafeCall(cuLaunchGrid( transform, width / block_size, height / block_size ));

    cutilDrvSafeCall( cuCtxSynchronize() );
    cutilCheckError( cutStopTimer( timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
    printf("%.2f Mpixels/sec\n", (width*height / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);
    cutilCheckError( cutDeleteTimer( timer));

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( size);
    // copy result from device to host
    cutilDrvSafeCall( cuMemcpyDtoH( h_odata, d_data, size) );

    // write result to file
    char output_filename[1024];
    strcpy(output_filename, image_path);
    strcpy(output_filename + strlen(image_path) - 4, "_out.pgm");
    cutilCheckError( cutSavePGMf( output_filename, h_odata, width, height));
    printf("Wrote '%s'\n", output_filename);

    // write regression file if necessary
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat", h_odata, width*height, 0.0));
    } 
    else 
    {
        // We need to reload the data from disk, because it is inverted upon output
        cutilCheckError( cutLoadPGMf(output_filename, &h_odata, &width, &height));

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", output_filename);
        printf("\treference: <%s>\n", ref_path);
        CUTBoolean res = cutComparefe( h_odata, h_data_ref, width*height, MIN_EPSILON_ERROR );
        printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    cutilDrvSafeCall(cuMemFree(d_data));
    cutilDrvSafeCall(cuArrayDestroy(cu_array));
//    free(h_data);
//    free(h_data_ref);
//    free(h_odata);
    cutFree(image_path);
    cutFree(ref_path);

    cutilDrvSafeCall(cuCtxDetach(cuContext));

    // If we are doing the QAtest, we quite without prompting
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "qatest") ||
        cutCheckCmdLineFlag( argc, (const char**) argv, "noprompt"))
    {
        exit(0);
    }

    cutilExit(argc, argv);
}

bool inline
findModulePath(const char *module_file, char **module_path, char **argv)
{
    *module_path = cutFindFilePath(module_file, argv[0]);
    if (module_path == 0) {
       printf("> findModulePath could not find file: <%s> \n", module_file); 
       return false;
    } else {
       printf("> findModulePath found file at <%s>\n", *module_path);
       return true;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! This initializes CUDA, and loads the *.ptx CUDA module containing the
//! kernel function.  After the module is loaded, cuModuleGetFunction 
//! retrieves the CUDA function pointer "cuFunction" 
////////////////////////////////////////////////////////////////////////////////
static CUresult
initCUDA(int argc, char **argv, CUfunction* transform)
{
    CUfunction cuFunction = 0;
    char* module_path;

    cutilDeviceInitDrv(cuDevice, argc, argv);

    CUresult status = cuCtxCreate( &cuContext, 0, cuDevice );
    if ( CUDA_SUCCESS != status )
        goto Error;

    // first search for the module_path before we try to load the results
    if (!findModulePath ("simpleTexture_kernel.ptx", &module_path, argv)) {
       if (!findModulePath ("simpleTexture_kernel.cubin", &module_path, argv)) {
          printf("> findModulePath could not find <simpleTexture_kernel> ptx or cubin\n");
          status = CUDA_ERROR_NOT_FOUND;
          goto Error;
       }
    } else {
       printf("> initCUDA loading module: <%s>\n", module_path);
    }

    status = cuModuleLoad(&cuModule, module_path);
    cutFree(module_path);
    if ( CUDA_SUCCESS != status ) {
        goto Error;
    }

    status = cuModuleGetFunction( &cuFunction, cuModule, "transformKernel" );
    if ( CUDA_SUCCESS != status )
        goto Error;
    *transform = cuFunction;
    return CUDA_SUCCESS;
Error:
    cuCtxDetach(cuContext);
    return status;
}
