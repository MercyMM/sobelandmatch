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


#define WIDTH 320
#define HEIGH 240
//#define HEIGH 120

using namespace std;
using namespace cv;

extern void* HostMal(void **p, long size);
extern void initCudaMalloc();

int main(int argc, char** argv)
{

    int32_t D_can_width = 60;  //[15,310] => 60
    int32_t D_can_height = 48; //[5, 230] => 46

    Elas::parameters param;//Æ¥Åä²ÎÊý
    param.postprocess_only_left = true;
//    Elas elas(param);
    Elas elas(param, (int32_t)WIDTH, (int32_t)HEIGH, D_can_width, D_can_height );

//    initCudaMalloc();
//    printf("initcudamalloc over\n");


    const int32_t dims[3] = {WIDTH,HEIGH,WIDTH}; // bytes per line = width
    float *D1_data, *D2_data, *D1_data_g, *D2_data_g;

//    D1_data = (float*)malloc(WIDTH*HEIGH*sizeof(float));
//    D2_data = (float*)malloc(WIDTH*HEIGH*sizeof(float));
    D1_data_g = (float*)HostMal((void**)&D1_data, WIDTH*HEIGH*sizeof(float));
    D2_data_g = (float*)HostMal((void**)&D2_data, WIDTH*HEIGH*sizeof(float));


    // init disparity image to -10


        for (int32_t i = 0; i < WIDTH*HEIGH; i++) {
            *(D1_data + i) = -10;
            *(D2_data + i) = -10;
        }

//    cvNamedWindow("capture_left");
//    cvNamedWindow("capture_right");
//    cvNamedWindow("capture_depth");
//    CvCapture* capture_left = cvCreateFileCapture("./video/result_left.avi");
//    CvCapture* capture_right = cvCreateFileCapture("./video/result_right.avi");

    struct timeval start, end;


//    IplImage* img1 = cvLoadImage("left4.png",0);
//    IplImage* img2 = cvLoadImage("right4.png",0);
    IplImage *img1 , *img2 ;
    char key ;
    uint8_t *I1, *I2;
//    printf("aaaaaa\n");
//    while(1)
//    {

//        img1 = cvQueryFrame(capture_left);
//        img2 = cvQueryFrame(capture_right);

//        img1 = cvLoadImage("left0.png",0);
//        img2 = cvLoadImage("right0.png",0);

//        cout<<"width,heigth,imageSize "<<img1->width<< "  "<< img1->height<< "  "<<img1->imageSize<< "  " << img1->widthStep<<endl;

//        cvShowImage("capture_left", img1);
//        cvShowImage("capture_right", img2);
//        key = cvWaitKey();

//        if('c' == key)
//        {
//            cvSaveImage("name_left.png", img1);
//            cvSaveImage("name_right.png", img2);
            img1 = cvLoadImage("left0.png",0);
            img2 = cvLoadImage("right0.png",0);
            I1 = (uint8_t*)img1->imageData;
            I2 = (uint8_t*)img2->imageData;

            gettimeofday(&start, NULL);
            elas.process(I1, I2, elas.D1_data_g, elas.D2_data_g, dims);
            gettimeofday(&end, NULL);
            double timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
            printf("elas process use : %fms\n", timeuse/1000);

            cout<<"show picture" << endl;
            IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);

            for (int32_t i=0; i<WIDTH*HEIGH; i++)
            {
//                img1f->imageData[i] = (uint8_t)max(255.0*(D1_data[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);
                img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);

            }
            cvShowImage("capture_dis1",img1f);

//            cvSaveImage("dis1.png", img1f);


            Mat imgMat(img1f, 0);
//            Mat imgMat(img1, 0);
            Mat imgColor;
            double vmin, vmax, alpha;
            minMaxLoc(imgMat, &vmin, &vmax);
            printf("min,max: %lf, %lf\n", vmin, vmax);
            alpha = 255.0 / (vmax - vmin);
            imgMat.convertTo(imgMat, CV_8U, alpha, -vmin*alpha);
            applyColorMap(imgMat, imgColor, COLORMAP_JET);
            imshow("capture_dis", imgColor);
//             cvShowImage("capture_depth",img1f);
//            cvSaveImage("aaa.jpg", imgColor);
//            imwrite("aaa.jpg", imgColor);
             cvWaitKey();





             img1 = cvLoadImage("left1.png",0);
             img2 = cvLoadImage("right1.png",0);
             I1 = (uint8_t*)img1->imageData;
             I2 = (uint8_t*)img2->imageData;

             gettimeofday(&start, NULL);
             elas.process(I1, I2, elas.D1_data_g, elas.D2_data_g, dims);
             gettimeofday(&end, NULL);
             timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
             printf("elas process use : %fms\n", timeuse/1000);

             cout<<"show picture" << endl;
//             IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);

             for (int32_t i=0; i<WIDTH*HEIGH; i++)
             {
 //                img1f->imageData[i] = (uint8_t)max(255.0*(D1_data[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);
                 img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);

             }
             cvShowImage("capture_dis1",img1f);

 //            cvSaveImage("dis1.png", img1f);


             Mat imgMat2(img1f, 0);
 //            Mat imgMat(img1, 0);
             Mat imgColor2;
//             double vmin, vmax, alpha;
             minMaxLoc(imgMat2, &vmin, &vmax);
             printf("min,max: %lf, %lf\n", vmin, vmax);
             alpha = 255.0 / (vmax - vmin);
             imgMat2.convertTo(imgMat2, CV_8U, alpha, -vmin*alpha);
             applyColorMap(imgMat2, imgColor2, COLORMAP_JET);
             imshow("capture_dis", imgColor2);
 //             cvShowImage("capture_depth",img1f);
 //            cvSaveImage("aaa.jpg", imgColor);
 //            imwrite("aaa.jpg", imgColor);
              cvWaitKey();

















//        }else if('n' == key)
//        {
//            printf("next\n");
//            continue;
//        }else if('q' == key)
//        {
//            break;
//        }
//         cvSaveImage("d2.png", img1f);
//    }

//    cvReleaseCapture(&capture_left);
//    cvReleaseCapture(&capture_right);
//    cvDestroyWindow("capture_left");
//    cvDestroyWindow("capture_right");
	return 0;
}
