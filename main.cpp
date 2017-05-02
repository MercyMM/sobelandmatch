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



typedef struct 
{
	float theta;
	float x;
	float y;
}speed;

struct point
{
    float x;
    float y;
} ;
typedef struct point point;


typedef struct 
{
	point p;
}PointStat;

typedef struct 
{
	float heading;
	float dist;
	float Vx;

}eval_t;


int DynamicWindowApproach(point* obstacle, int obs_nums,  point goal);
int GenerateTraj(point* obstacle, int obs_nums);
int NormalizeEval(point goal);
int Evaluation();

//制动距离
//65个方向
//预测四秒后的位置
#define STOPDIST 1
#define DIRECTIONS 17
#define PREDICTT 4 

#define DEL_DIS		0.5
#define SAFE_DIS	0.5

//eval func param
#define HEADING 0.05
#define DIST	0.2
#define VEL		0.1

speed SpeedWindow[] = 
{
   {64, 0.44, 0.90},
   {56, 0.56, 0.83},
   {48, 0.67, 0.74},
   {40, 0.77, 0.64},
   {32, 0.85, 0.53},
   {24, 0.91, 0.41},
   {16, 0.96, 0.28},
   {8, 0.99, 0.14},
   {0, 1.00, 0.00},
   {-8, 0.99, -0.14},
   {-16, 0.96, -0.28},
   {-24, 0.91, -0.41},
   {-32, 0.85, -0.53},
   {-40, 0.77, -0.64},
   {-48, 0.67, -0.74},
   {-56, 0.56, -0.83},
   {-64, 0.44, -0.90}
};

point predict[DIRECTIONS];
eval_t EvalDB[DIRECTIONS];	
eval_t EvalDB_Nor[DIRECTIONS];



extern void* HostMal(void **p, long size);
extern void initCudaMalloc();
extern void SetDeviceMap();
extern void allocFreeCount();

int main(int argc, char** argv)
{

    int32_t D_can_width = 60;  //[15,310] => 60
    int32_t D_can_height = 48; //[5, 230] => 46

    SetDeviceMap();
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
        img1 = cvLoadImage("./picture/left_58.png",0);
        img2 = cvLoadImage("./picture/right_58.png",0);

//        cout<<"width,heigth,imageSize "<<img1->width<< "  "<< img1->height<< "  "<<img1->imageSize<< "  " << img1->widthStep<<endl;

 //       cvShowImage("capture_left", img1);
//        cvShowImage("capture_right", img2);
//        key = cvWaitKey();



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
            printf("xxxxxxxxxxxxxxxxxxxxxxx elas process use : %fms\n", timeuse/1000);

            IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
            IplImage* img2f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
            IplImage* img3f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
            for (int32_t i=0; i<WIDTH*HEIGH; i++){
                float dis = elas.D1_data_c[i];
                img1f->imageData[i] = (uint8_t)max(255.0*(dis)/63, 0.0);
            }
            for (int32_t i=0; i<WIDTH*HEIGH; i++){
                float dis = elas.cloud_c[i].z;
                if ( 10000 < dis)
                    dis = 10000;
                if(dis <= 0)
                    img2f->imageData[i] = 255;
                else
                    img2f->imageData[i] = (uint8_t)(dis / 40);
            }


//            for(int i = 0; i < WIDTH * HEIGH / 5; i += 5){
//                float dis = elas.D1_data_c[i * 5];
//                if ( 10000 < dis)
//                    dis = 10000;
//                img3f->imageData[i] = (uint8_t)(dis / 40);
//            }
	 cvShowImage("capture_left", img1);
            cvShowImage("capture_dis1",img1f);
            cvShowImage("consistency",img2f);
//            cvShowImage("non_consistency",img3f);
//cvSaveImage("dist3.png", img3f);
//cvSaveImage("dist3.png", img2f);

            Mat imgMat(img1f, 0);
            Mat imgColor;
            minMaxLoc(imgMat, &vmin, &vmax);
            alpha = 255.0 / (vmax - vmin);
            imgMat.convertTo(imgMat, CV_8U, alpha, -vmin*alpha);
            applyColorMap(imgMat, imgColor, COLORMAP_JET);
            imshow("capture_dis", imgColor);


//            for(int i = 320 * 200 + 20; i < 320 * 200 +30 ; i ++)
//            {
//                printf("(%f,", elas.cloud_c[i].x / 1000);
//                printf("%f) ", elas.cloud_c[i].z / 1000);
//            }
//            cout<<endl;
//            return 0;



        gettimeofday(&start, NULL);
        //loop 5 lines;
        vector<struct point> obstacle;
        for(int i = 320 * 120 + 20; i < 320 * 121 - 20; i++){
            float dis = elas.cloud_c[i].z;
            int num = 1;
            float obs_z;
            float obs_x;
            int j = i;
            if(dis <= 0)
                continue;
            if(dis < 10000 && dis > 0){
                obs_z = dis;
                obs_x = elas.cloud_c[j].x;
                for( j += 1; j < i + 5; j++){
                    float dis2 = elas.cloud_c[j].z;
                    if(dis2 < dis && dis2 > 0){
                        obs_z = dis2;
                        obs_x = elas.cloud_c[j].x;
                    }
                    //printf("%d, %f\n", j - 320*120, dis2);
                    if( dis2 < 10000 && dis2 > 0)
                        num++;
                }
                //printf("num = %d\n", num);
            }
            if(num > 3){
                struct point Ob;
                Ob.x = obs_z / 1000;
                Ob.y = obs_x / 1000;
                //                    printf("push: %d, %f, %f\n", obs_x, Ob.x, Ob.y);
                obstacle.push_back(Ob);
                i = j - 1;
            }
        }
            printf("\n\n\n");
        vector<struct point>::iterator iter = obstacle.begin();
        for(; iter != obstacle.end(); iter++){
            printf("{%f, %f},\n", (*iter).x, (*iter).y);
        }

//return;




            gettimeofday(&end, NULL);
            timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
            printf("DWA : %fms\n", timeuse/1000);

            gettimeofday(&start, NULL);
    point goal = {20,0};
    int index = DynamicWindowApproach( &obstacle[0], obstacle.size(), goal);

    printf("index = %d\n", index);

            gettimeofday(&end, NULL);
            timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
            printf("DWA : %fms\n", timeuse/1000);



        key =  cvWaitKey(0);
        if( 'q' == key){
            break;
        }else if( ' ' == key){
            key = getchar();
        }

//    usleep(5000000);
//#ifdef LOAD_PIC
    }
//#endif
return 0;
}





int DynamicWindowApproach(point* obstacle, int obs_nums,  point goal)
{
	//1. compute window
	//2. 模拟窗口内的运动轨迹 ,64条轨迹,存入traj向量
	//traj 存放64个移动4s后的飞行器位置
	GenerateTraj(obstacle, obs_nums);

	printf("NormalizeEval\n");
	// 各评价函数正则化
	NormalizeEval(goal);

	//3. 评价函数选择最优路线
	return  Evaluation();

	//4. move dt ms.初期设想，运行20ｍｓ
	// MoveDt();
}

int GenerateTraj(point* obstacle, int obs_nums)
{

	float Vx, Vy;
	float Px, Py;


	printf("predict point:\n");
	for(int i = 0; i < DIRECTIONS; i++)
	{
		Vx = SpeedWindow[i].x;
		Vy = SpeedWindow[i].y;

		float obs_dis = 100;
		int flag = 0;
		//如果轨迹穿过障碍物，或很接近障碍物，则舍去该轨迹
		for(int t = 0; t < PREDICTT/DEL_DIS; t++)
		{

			Px = Vx * (t * DEL_DIS);
			Py = Vy * (t * DEL_DIS);
			for(int k = 0; k < obs_nums; k++)
			{
				obs_dis = pow( pow(Px - obstacle[k].x, 2) + \
						pow(Py - obstacle[k].y, 2), 0.5);
				if(obs_dis < SAFE_DIS )
				{
					flag = 1;
					predict[i].x = 0;
					predict[i].y = 0;
					break;
				}
			}
			if(1 == flag)
				break;
		}
		if(1 == flag)
		{
			printf("xxxx\n");
			continue;
		}

		Px = Vx * PREDICTT;
		Py = Vy * PREDICTT;
		predict[i].x = Px;
		predict[i].y = Py;
		// predict[i].p.x = SpeedWindow[i].x * PREDICTT;
		// predict[i].p.y = SpeedWindow[i].y * PREDICTT;
		printf("%f, %f  \n",Px, Py);

	}
	printf("\n");
}




int NormalizeEval(point goal)
{
	for(int i = 0; i < DIRECTIONS; i ++)
	{
		if( 0 == predict[i].x && 0 == predict[i].y)
		{
//			printf("xxxx\n");
			continue;
		}
		float del_x = goal.x-predict[i].x;
		if ( 0 == del_x)
		{
//			printf("del_x == 0\n");
			while(1);
		}
		
		EvalDB[i].heading = (goal.y-predict[i].y) / (goal.x-predict[i].x)   ;
		EvalDB[i].dist = pow( pow(goal.x-predict[i].x, 2) + pow(goal.y-predict[i].y, 2), 0.5);
		EvalDB[i].Vx = SpeedWindow[i].x;

		if( 0 > EvalDB[i].heading)
			EvalDB[i].heading = -EvalDB[i].heading;
//		printf(" %f\n", EvalDB[i].heading);
	}
	float heading_sum, dist_sum, Vx_sum;
	for(int i = 0; i < DIRECTIONS; i++)
	{
		if( 0 == predict[i].x && 0 == predict[i].y)
		{
			continue;
		}
		heading_sum += EvalDB[i].heading;
		dist_sum 	+= EvalDB[i].dist;
		Vx_sum		+= EvalDB[i].Vx;
	}
//	printf("heading_sum: %f\n", heading_sum);
	// printf("%f, %f, %f\n", heading_sum, dist_sum, Vx_sum);
	for(int i = 0; i < DIRECTIONS; i++)
	{
		if( 0 == predict[i].x && 0 == predict[i].y)
		{
            printf("xxxx\n");
			continue;
		}
		EvalDB_Nor[i].heading = EvalDB[i].heading / heading_sum;
		EvalDB_Nor[i].dist 	  = EvalDB[i].dist /dist_sum;
		EvalDB_Nor[i].Vx      = EvalDB[i].Vx / Vx_sum;
        printf("%f, %f, %f\n", EvalDB_Nor[i].heading,EvalDB_Nor[i].dist,EvalDB_Nor[i].Vx);
	}

}


int Evaluation()
{

	int index = -1;
	float eval = 100;;
	for(int i = 0; i < DIRECTIONS; i++)
	{
		if( 0 == predict[i].x && 0 == predict[i].y)
			continue;
		float heading 	= EvalDB_Nor[i].heading;
		float dist 		= EvalDB_Nor[i].dist;
		float Vx 		= EvalDB_Nor[i].Vx;

		float eval_tmp = HEADING * heading + DIST * dist + VEL * Vx;
		if(eval > eval_tmp)
		{
			eval = eval_tmp;
			index = i;
		}
	}
	printf("index = %d\n", index);
	return index;
}















