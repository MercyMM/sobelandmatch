#include <stdio.h>
// #include <tchar.h>

#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include "elas.h"
#include <sstream>
#include <iomanip>
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <string.h>

cl_context CreateContext();

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);


///
//  Cleanup any created OpenCL resources
//
void clCleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem imageObjects[2],
	cl_sampler sampler);

///
//  Load an image using the FreeImage library and create an OpenCL
//  image out of it
//
cl_mem LoadImage(cl_context context, char *fileName, int &width, int &height);

///
//  Save an image using the FreeImage library
//
/*
bool SaveImage(char *fileName, char *buffer, int width, int height)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
	FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)buffer, width,
		height, width * 4, 32,
		0xFF000000, 0x00FF0000, 0x0000FF00);
	return FreeImage_Save(format, image, fileName);
}
*/
///
//  Round up to the nearest multiple of the group size
//
size_t RoundUp(int groupSize, int globalSize);