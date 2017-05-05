#include <stdio.h>
#include <unistd.h>
#include "opencv/cv.h"
#include "opencv/cxmisc.h"
#include "opencv/highgui.h"
#include "opencv/cvaux.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include "elas.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string.h>
#include <arm_neon.h>
#include <sys/time.h>


int main(int argc, char** argv)
{
    CvCapture* capture_left = cvCreateFileCapture("../video/result_left3.avi");
    CvCapture* capture_right = cvCreateFileCapture("../video/result_right3.avi");

    struct timeval start, end;
    double timeuse;

    IplImage *img1 , *img2 ;
    char key ;
    cvSetCaptureProperty(capture_left, CV_CAP_PROP_POS_FRAMES, 50);
    cvSetCaptureProperty(capture_right, CV_CAP_PROP_POS_FRAMES, 50);
//    printf("aaaaaa\n");
    string left_name, right_name;
    string left = "left";
    string right = "right";
    char name[30] ;
    int name_num = 0;
    while(1)
    {
        sprintf(name, "./picture3/left_%d.png", name_num);
        left_name = name;
        sprintf(name, "./picture3/right_%d.png", name_num);
        right_name = name;

        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        img1 = cvQueryFrame(capture_left);
        img2 = cvQueryFrame(capture_right);
        cvShowImage("picture", img1);

        printf("start\n");
        cvSaveImage(left_name.c_str(), img1);
        cvSaveImage(right_name.c_str(), img2);

        key = cvWaitKey(5000);


        if('t' == key){
            return;
        }
        name_num++;

    }
