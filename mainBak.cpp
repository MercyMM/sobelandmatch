#include <stdio.h>
// #include <tchar.h>
#include "opencv/cv.h"
#include "opencv/cxmisc.h"
#include "opencv/highgui.h"
#include "opencv/cvaux.h"
#include <vector>
#include <string>
//#include "clelas.h"
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include "elas.h"
#include <sstream>
#include <iomanip>
//#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <string.h>
//#include "cl_user.h"
#include <arm_neon.h>
#include <sys/time.h>
using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	//cl_context context = 0;
	//cl_command_queue commandQueue = 0;
	//cl_program program = 0;
	//cl_device_id device = 0;
	//cl_kernel kernel = 0;
	// cl_mem imageObjects[2] = { 0, 0 };
	// cl_sampler sampler = 0;
	// cl_int errNum;


	// clElas::parameters param;//匹配参数
 //    param.postprocess_only_left = true;
 //    clElas elas(param);
    Elas::parameters param;//匹配参数
    param.postprocess_only_left = true;
    Elas elas(param);


#define WEIGH 320
#define HEIGH 240

    const int32_t dims[3] = {WEIGH,HEIGH,WEIGH}; // bytes per line = width
    float* D1_data = (float*)malloc(WEIGH*HEIGH*sizeof(float));
    float* D2_data = (float*)malloc(WEIGH*HEIGH*sizeof(float));
    cvNamedWindow("capture_left");
    cvNamedWindow("capture_right");
    cvNamedWindow("capture_depth");
    CvCapture* capture_left = cvCreateFileCapture("./result_left.avi");
    CvCapture* capture_right = cvCreateFileCapture("./result_right.avi");




    IplImage* img1 = cvLoadImage("left4.png",0);
    IplImage* img2 = cvLoadImage("right4.png",0);


    uint8_t * I1 = (uint8_t*)img1->imageData;
	uint8_t * I2 = (uint8_t*)img2->imageData;


    cout<<"width,heigth,imageSize "<<img1->width<< "  "<< img1->height<< "  "<<img1->imageSize<< "  " << img1->widthStep<<endl;
    int dim = HEIGH;
struct timeval start, end;

gettimeofday(&start, NULL);

    elas.process(I1,I2,D1_data,D2_data,dims, dim);
gettimeofday(&end, NULL);
double timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
printf("elas process use : %fms\n", timeuse/1000);

    cout<<"show picture" << endl;

    IplImage* img1f = cvCreateImage(cvSize(WEIGH, HEIGH), IPL_DEPTH_8U,1);
	
    for (int32_t i=0; i<WEIGH*HEIGH; i++)
	{
		img1f->imageData[i] = (uint8_t)max(255.0*(D1_data[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);
	}
	//IplImage *dst = cvCreateImage(cvGetSize(img1f), IPL_DEPTH_8U, 3); 
     cvShowImage("Stereo_Match",img1f);
    waitKey(0);
     cvSaveImage("d2.png", img1f);

	return 0;
}
