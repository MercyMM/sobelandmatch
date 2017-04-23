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
#include "cl_user.h"

//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}
	std::cout<<"num of platform:"<<numPlatforms<<std::endl;
	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.

	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error in kernel: " << std::endl;
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		// Allocate memory for the log
		char *log = (char *) malloc(log_size);
		// Get the log
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		// Print the log
		printf("%s\n", log);
		// Determine the reason for the error
		
		while(1);
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}


///
//  Cleanup any created OpenCL resources
//
void clCleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem imageObjects[2],
	cl_sampler sampler)
{
//	for (int i = 0; i < 2; i++)
//	{
//		if (imageObjects[0] != 0)
//			clReleaseMemObject(imageObjects[0]);
//	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (sampler != 0)
		clReleaseSampler(sampler);

	if (context != 0)
		clReleaseContext(context);

}

///
//  Load an image using the FreeImage library and create an OpenCL
//  image out of it
//
//cl_mem LoadImage(cl_context context, char *fileName, int &width, int &height)
//{
//	cv::Mat image1 = cv::imread(fileName);
//	width = image1.cols;
//	height = image1.rows;
//	char *buffer = new char[width * height * 4];
//	int w = 0;
//	for (int v = height - 1; v >= 0; v--)
//	{
//		for (int u = 0; u < width;u++)
//		{
//			buffer[w++] = image1.at<cv::Vec3b>(v, u)[0];  
//            buffer[w++] = image1.at<cv::Vec3b>(v, u)[1];  
//            buffer[w++] = image1.at<cv::Vec3b>(v, u)[2]; 
//			w++;
//		}
//	}
//	
//
//	// Create OpenCL image
//	cl_image_format clImageFormat;
//	clImageFormat.image_channel_order = CL_RGBA;
//	clImageFormat.image_channel_data_type = CL_UNORM_INT8;
//
//	cl_int errNum;
//	cl_mem clImage;
//	clImage = clCreateImage2D(context,
//		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//		&clImageFormat,
//		width,
//		height,
//		0,
//		buffer,
//		&errNum);
//
//	if (errNum != CL_SUCCESS)
//	{
//		std::cerr << "Error creating CL image object" << std::endl;
//		return 0;
//	}
//
//	return clImage;
//}

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
size_t RoundUp(int groupSize, int globalSize)
{
	int r = globalSize % groupSize;
	if (r == 0)
	{
		return globalSize;
	}
	else
	{
		return globalSize + groupSize - r;
	}
}