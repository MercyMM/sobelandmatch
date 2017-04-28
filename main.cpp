#include <stdio.h>
#include <unistd.h>
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
    Elas elas(param, (int32_t)WIDTH, (int32_t)HEIGH, D_can_width, D_can_height );

//    initCudaMalloc();
//    printf("initcudamalloc over\n");


    const int32_t dims[3] = {WIDTH,HEIGH,WIDTH}; // bytes per line = width


//    cvNamedWindow("capture_left");
//    cvNamedWindow("capture_right");
//    cvNamedWindow("capture_depth");
    CvCapture* capture_left = cvCreateFileCapture("../video/result_left1.avi");
    CvCapture* capture_right = cvCreateFileCapture("../video/result_right1.avi");

    struct timeval start, end;
    double timeuse;
    double vmin, vmax, alpha;

//    IplImage* img1 = cvLoadImage("left4.png",0);
//    IplImage* img2 = cvLoadImage("right4.png",0);
    IplImage *img1 , *img2 ;
    IplImage *imgBak1, *imgBak2;
    imgBak1 = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U, 1);
    imgBak2 = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U, 1);
    char key ;
    uint8_t *I1, *I2;
//    cvSetCaptureProperty(capture_left, CV_CAP_PROP_POS_FRAMES, 220);
//    cvSetCaptureProperty(capture_right, CV_CAP_PROP_POS_FRAMES, 220);
//    printf("aaaaaa\n");
string left_name, right_name;
    string left = "left";
    string right = "right";//        cvSaveImage(left_name.c_str(), img1);
    //        cvSaveImage(right_name.c_str(), img2);
    //        printf("save\n");
    //        name_num++;
    char name[30] ;
    int name_num = 0;
    while(1)
    {
        sprintf(name, "./picture1/left_%d.png", name_num);
        left_name = name;
        sprintf(name, "./picture1/right_%d.png", name_num);
        right_name = name;
//#ifdef LOAD_PIC
//        img1 = cvLoadImage(left_name.c_str(), 0);
//        img2 = cvLoadImage(right_name.c_str(), 0);
        name_num++;
//        cvShowImage("capture_left", img1);
//#endif
//        cout<< left_name;
//        cout << right_name;
//        key = getchar();
//        name_num ++;
//#ifdef SAVE_PIC
//        img1 = cvQueryFrame(capture_left);
//        img2 = cvQueryFrame(capture_right);
//        cvShowImage("picture", img1);

//        key = cvWaitKey(5000);


//        if('s' == key){
//            cvSaveImage(left_name.c_str(), img1);
//            cvSaveImage(right_name.c_str(), img2);
//            name_num++;
//        }else if('t' == key){
//            return;
//        }
//        printf("start\n");
//        cvSaveImage(left_name.c_str(), img1);
//        cvSaveImage(right_name.c_str(), img2);
//        printf("save\n");
//        name_num++;

//    }
//#endif
//        sleep(1);
//        printf("load\n");
//        img1 = cvLoadImage("a1-temp-l.png", 0);
//        img2 = cvLoadImage("a1-temp-r.png", 0);

//            key =  cvWaitKey(0);
//            if(key == 'q')
//                return;
//    }
        img1 = cvLoadImage("left0.png",0);
        img2 = cvLoadImage("right0.png",0);

//        cout<<"width,heigth,imageSize "<<img1->width<< "  "<< img1->height<< "  "<<img1->imageSize<< "  " << img1->widthStep<<endl;

        cvShowImage("capture_left", img1);
//        cvShowImage("capture_right", img2);
//        key = cvWaitKey();

//        float mm = 1234.1234;
//        int mm_i = (int)mm;
//        printf("%d\n", mm_i);
//        return;

//        if('c' == key)
//        {
//            cvSaveImage("name_left.png", img1);
//            cvSaveImage("name_right.png", img2);

//            img1 = cvLoadImage("left0.png",0);
//            img2 = cvLoadImage("right0.png",0);
//        for(int i = 0; i < WIDTH * HEIGH; i++){
//            *(imgBak1->imageData + i) = *(img1->imageData + i);
//            *(imgBak2->imageData + i) = *(img2->imageData + i);
//        }
            I1 = (uint8_t*)img1->imageData;
            I2 = (uint8_t*)img2->imageData;
//        I1 = (uint8_t*)imgBak1->imageData;
//        I2 = (uint8_t*)imgBak2->imageData;
//                    cvShowImage("capture_dis222",imgBak1);
            gettimeofday(&start, NULL);
            elas.process(I1, I2);
//            elas.process(imgBak1, imgBak2);
            gettimeofday(&end, NULL);
            timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
            printf("elas process use : %fms\n", timeuse/1000);

            IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
            IplImage* img2f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
            for (int32_t i=0; i<WIDTH*HEIGH; i++)
            {
//                img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i*3 + 2]-param.disp_min)/(param.disp_max-param.disp_min),0.0);

                img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i])/63, 0.0);
                img2f->imageData[i] = (uint8_t)(max((int)(elas.D1_data_c[i]), 0) / 1000) ;
            }
            cout<<endl;
            for(int v = 120; v < 240; v++){
                printf("height = %d\n", v);
                for(int u = 0; u < 320; u += 2){
                    if( -1 == elas.D1_data_c[ u + v * 320 ])
                        continue;
                    printf("%d ", (int)elas.D1_data_c[u + v * 320] / 1000);
                }
                cout<<endl;

            }
            cout<<endl;
            cvShowImage("capture_dis1",img1f);
            cvShowImage("capture_dis2",img2f);
cvSaveImage("dist1.png", img1f);
cvSaveImage("dist2.png", img2f);

            Mat imgMat(img1f, 0);
            Mat imgColor;
            minMaxLoc(imgMat, &vmin, &vmax);
            alpha = 255.0 / (vmax - vmin);
            imgMat.convertTo(imgMat, CV_8U, alpha, -vmin*alpha);
            applyColorMap(imgMat, imgColor, COLORMAP_JET);
            imshow("capture_dis", imgColor);
            key =  cvWaitKey(0);
            if( 'q' == key){
                return;
//                break;
            }else if( ' ' == key){
                key = getchar();
            }

//    usleep(5000000);
//#ifdef LOAD_PIC
    }
//#endif
return 0;
}

//            img1 = cvLoadImage("left1.png",0);
//            img2 = cvLoadImage("right1.png",0);
//            I1 = (uint8_t*)img1->imageData;
//            I2 = (uint8_t*)img2->imageData;

//            gettimeofday(&start, NULL);
//            elas.process(I1, I2);
//            gettimeofday(&end, NULL);
//            timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
//            printf("elas process use : %fms\n", timeuse/1000);

////            IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
//            for (int32_t i=0; i<WIDTH*HEIGH; i++)
//            {
//                img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i*3 + 2]-param.disp_min)/(param.disp_max-param.disp_min),0.0);
//            }
//            cvShowImage("capture_dis1",img1f);
//cvShowImage("picture", img1);
//            Mat imgMat1(img1f, 0);
//            Mat imgColor1;
//            minMaxLoc(imgMat1, &vmin, &vmax);
//            alpha = 255.0 / (vmax - vmin);
//            imgMat1.convertTo(imgMat1, CV_8U, alpha, -vmin*alpha);
//            applyColorMap(imgMat1, imgColor1, COLORMAP_JET);
//            imshow("capture_dis", imgColor1);
//            key =  cvWaitKey(0);
//            if( 'q' == key){
//                break;
//            }








//             img1 = cvLoadImage("left1.png",0);
//             img2 = cvLoadImage("right1.png",0);
//             I1 = (uint8_t*)img1->imageData;
//             I2 = (uint8_t*)img2->imageData;

//             gettimeofday(&start, NULL);
//             elas.process(I1, I2, elas.D1_data_g, elas.D2_data_g, dims);
//             gettimeofday(&end, NULL);
//             timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
//             printf("elas process use : %fms\n", timeuse/1000);

////             IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);

//             for (int32_t i=0; i<WIDTH*HEIGH; i++)
//             {
// //                img1f->imageData[i] = (uint8_t)max(255.0*(D1_data[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);
//                 img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);

//             }
//             cvShowImage("capture_dis1",img1f);

// //            cvSaveImage("dis1.png", img1f);


//             Mat imgMat2(img1f, 0);
// //            Mat imgMat(img1, 0);
//             Mat imgColor2;
////             double vmin, vmax, alpha;
//             minMaxLoc(imgMat2, &vmin, &vmax);
////             printf("min,max: %lf, %lf\n", vmin, vmax);
//             alpha = 255.0 / (vmax - vmin);
//             imgMat2.convertTo(imgMat2, CV_8U, alpha, -vmin*alpha);
//             applyColorMap(imgMat2, imgColor2, COLORMAP_JET);
//             imshow("capture_dis", imgColor2);
// //             cvShowImage("capture_depth",img1f);
// //            cvSaveImage("aaa.jpg", imgColor);
// //            imwrite("aaa.jpg", imgColor);
//              cvWaitKey();





//              img1 = cvLoadImage("left4.png",0);
//              img2 = cvLoadImage("right4.png",0);
//              I1 = (uint8_t*)img1->imageData;
//              I2 = (uint8_t*)img2->imageData;

//              gettimeofday(&start, NULL);
//              elas.process(I1, I2, elas.D1_data_g, elas.D2_data_g, dims);
//              gettimeofday(&end, NULL);
//              timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
//              printf("elas process use : %fms\n", timeuse/1000);

////              cout<<"show picture" << endl;

//              for (int32_t i=0; i<WIDTH*HEIGH; i++)
//              {
//                  img1f->imageData[i] = (uint8_t)max(255.0*(elas.D1_data_c[i]-param.disp_min)/(param.disp_max-param.disp_min),0.0);

//              }
//              cvShowImage("capture_dis1",img1f);

//              Mat imgMat3(img1f, 0);
//              Mat imgColor3;
// //             double vmin, vmax, alpha;
//              minMaxLoc(imgMat3, &vmin, &vmax);
////              printf("min,max: %lf, %lf\n", vmin, vmax);
//              alpha = 255.0 / (vmax - vmin);
//              imgMat3.convertTo(imgMat3, CV_8U, alpha, -vmin*alpha);
//              applyColorMap(imgMat3, imgColor3, COLORMAP_JET);
//              imshow("capture_dis", imgColor3);
//  //             cvShowImage("capture_depth",img1f);
//  //            cvSaveImage("aaa.jpg", imgColor);
//  //            imwrite("aaa.jpg", imgColor);
//               cvWaitKey();













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

