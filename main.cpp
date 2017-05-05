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


#define WIDTH 320
#define HEIGH 240

using namespace std;
using namespace cv;


typedef struct 
{
	float theta;
	float x;
	float y;
}speed;

struct obs_point
{
    float x;
    float y;
    int num;
};

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
#define PREDICTT 6      //predict time 6s;

#define DEL_DIS		0.5
#define SAFE_DIS	0.7    //M100: width 0.51; 1m is too far
//use ./picture3/left18 will no path

//eval func param
#define HEADING 0.05
#define DIST	0.2
#define VEL		0.1

//M100 speed is 0.5m/s
speed SpeedWindow[] =
{
    {58, 0.42402, 0.26497}, // 0
    {62, 0.44147, 0.23475}, // 1
    {66, 0.45677, 0.20338}, // 2
    {70, 0.46984, 0.17103}, // 3
    {74, 0.48063, 0.13784}, // 4
    {78, 0.48907, 0.10398}, // 5
    {82, 0.49513, 0.06961}, // 6
    {86, 0.49878, 0.03490}, // 7
    {90, 0.50000, 0.00002}, // 8
    {94, 0.49878, -0.03485}, // 9
    {98, 0.49514, -0.06956}, // 10
    {102, 0.48908, -0.10393}, // 11
    {106, 0.48064, -0.13779}, // 12
    {110, 0.46986, -0.17098}, // 13
    {114, 0.45678, -0.20334}, // 14
    {118, 0.44149, -0.23471}, // 15
    {122, 0.42404, -0.26493} // 16

};

point predict[DIRECTIONS];
eval_t EvalDB[DIRECTIONS];	
eval_t EvalDB_Nor[DIRECTIONS];


void createObs(Elas &elas, struct obs_point *obs_arr, \
               vector<struct point> *obstacle);

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

    const int32_t dims[3] = {WIDTH,HEIGH,WIDTH}; // bytes per line = width

    CvCapture* capture_left = cvCreateFileCapture("../video/result_left3.avi");
    CvCapture* capture_right = cvCreateFileCapture("../video/result_right3.avi");

    struct timeval start, end;
    double timeuse;
    double vmin, vmax, alpha;
    IplImage *img1 , *img2 ;
    IplImage *imgBak1, *imgBak2;
    imgBak1 = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U, 1);
    imgBak2 = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U, 1);
    char key ;
    uint8_t *I1, *I2;
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
                img1 = cvLoadImage("./picture3/left_18.png",0);
                img2 = cvLoadImage("./picture3/right_18.png",0);
//        img1 = cvLoadImage("left0.png",0);
 //       img2 = cvLoadImage("right0.png",0);
    //    cvShowImage("x",img1);

        I1 = (uint8_t*)img1->imageData;
        I2 = (uint8_t*)img2->imageData;

        gettimeofday(&start, NULL);
        elas.process(I1, I2);
        gettimeofday(&end, NULL);
        timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
        printf("xxxxxxxxxxxxxxxxxxxxxxx elas process use : %fms\n", timeuse/1000);

        struct obs_point obs_arr[320] = {0};
        vector<struct point> obstacle;
        gettimeofday(&start, NULL);
        createObs(elas, obs_arr, &obstacle);
        gettimeofday(&end, NULL);
        timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
        printf("createObs : %fms\n", timeuse/1000);



        gettimeofday(&start, NULL);
        point goal = {20,-1};
        int index = DynamicWindowApproach( &obstacle[0], obstacle.size(), goal);
        gettimeofday(&end, NULL);
        timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
        printf("DWA : %fms\n", timeuse/1000);

return 0;
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

//	printf("NormalizeEval\n");
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


//	printf("predict point:\n");
	for(int i = 0; i < DIRECTIONS; i++)
	{
		Vx = SpeedWindow[i].x;
		Vy = SpeedWindow[i].y;

		float obs_dis = 100;
		int flag = 0;
		//如果轨迹穿过障碍物，或很接近障碍物，则舍去该轨迹
        for(int t = 0; t < PREDICTT/DEL_DIS; t++) //t = (0, 11)
		{

			Px = Vx * (t * DEL_DIS);
			Py = Vy * (t * DEL_DIS);
			for(int k = 0; k < obs_nums; k++)
			{
				obs_dis = pow( pow(Px - obstacle[k].x, 2) + \
						pow(Py - obstacle[k].y, 2), 0.5);
                if(15 == i)
                {
//                    printf("15 dis: %d, %f\n", t, obs_dis);
                }
				if(obs_dis < SAFE_DIS )
                {
//                    printf("%d, %d, %d, (%f, %f), %f\n", i, t, k, \
//                           Px, Py, obs_dis);
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
//			printf("xxxx\n");
			continue;
		}

		Px = Vx * PREDICTT;
		Py = Vy * PREDICTT;
		predict[i].x = Px;
		predict[i].y = Py;
		// predict[i].p.x = SpeedWindow[i].x * PREDICTT;
		// predict[i].p.y = SpeedWindow[i].y * PREDICTT;
//		printf("%f, %f  \n",Px, Py);

	}
//	printf("\n");
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
//        printf("%f, %f, %f\n", EvalDB_Nor[i].heading,EvalDB_Nor[i].dist,EvalDB_Nor[i].Vx);
        printf("%d: %f, %f  =  %f\n", i, EvalDB_Nor[i].heading, EvalDB_Nor[i].dist, \
               HEADING * EvalDB_Nor[i].heading + DIST * EvalDB_Nor[i].dist);
	}

}

int Evaluation()
{

	int index = -1;
    float eval = 100;
	for(int i = 0; i < DIRECTIONS; i++)
	{
		if( 0 == predict[i].x && 0 == predict[i].y)
			continue;
		float heading 	= EvalDB_Nor[i].heading;
		float dist 		= EvalDB_Nor[i].dist;
		float Vx 		= EvalDB_Nor[i].Vx;

//		float eval_tmp = HEADING * heading + DIST * dist + VEL * Vx;
        float eval_tmp = HEADING * heading + DIST * dist ;
		if(eval > eval_tmp)
		{
			eval = eval_tmp;
			index = i;
        }
	}
	printf("index = %d\n", index);
	return index;
}


void createObs(Elas &elas, struct obs_point *obs_arr, \
               vector<struct point> *obstacle)
{
    struct timeval start, end;
    double timeuse;



    IplImage* img1f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
    IplImage* img2f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);
    IplImage* img3f = cvCreateImage(cvSize(WIDTH, HEIGH), IPL_DEPTH_8U,1);

    for (int32_t i=0; i<WIDTH*HEIGH; i++){
        float dis_y = elas.cloud_c[i].y;
        if ( 1000 < dis_y)
            dis_y = 1000;
        if(dis_y <= -1000)
            dis_y = -1000;
        dis_y += 1000;
        dis_y /= 10;

            img1f->imageData[i] = (uint8_t)(dis_y);
    }


    //x axi is cloud_c.z
    for (int32_t i=0; i<WIDTH*HEIGH; i++){
        float dis = elas.cloud_c[i].z;
        if ( 10000 < dis)
            dis = 10000;
        if(dis <= 0)
            img2f->imageData[i] = 255;
        else
            img2f->imageData[i] = (uint8_t)(dis / 40);
    }
    gettimeofday(&start, NULL);

    //for(int v = 19; v < 220; v ++ )
    for(int v = 80; v < 140; v ++ )
    {
        for(int u = 3; u < 313; u ++)
        {
            int index = u + v * WIDTH;
            float dis_y = elas.cloud_c[index].y;
            float dis_z = elas.cloud_c[index].z;
            float dis_x = elas.cloud_c[index].x;
            img3f->imageData[index] = 255;

            if( dis_y > -200 && dis_y < 200 )
            {
                if ( 10000 < dis_z)
                    dis_z = 10000;
                if(dis_z <= 0)
                    img3f->imageData[index] = 255;
                else
                    img3f->imageData[index] = (uint8_t)(dis_z / 40);
                if(dis_z < 6000 && dis_z > 0)
                {
                    //add this point to obstacle
                    struct obs_point *p = &obs_arr[u];

                    //if(124 == u)
                    //    printf("124: (%f, %f) \n", dis_z, dis_x);
                    if(0 == p->y)
                    {
                        p->y = dis_x;
                    }else
                    {
                        p->y = (p->y + dis_x)/2;
                    }
                    if(0 == p->x )
                    {
                        p->x = dis_z;
                    }else
                    {
                        p->x = min(dis_z , p->x);
                    }
                    p->num++;
                }else
                    continue;

            }else
                continue;
        }

    }

    //clear not use
    for(int i = 0; i < 320; i++)
    {
        if(obs_arr[i].num >= 5)
        {
//           printf("(%f, %f) %d %d\n", obs_arr[i].x, obs_arr[i].y, obs_arr[i].num, i);

        }
        else //if(18 > obs_arr[i].num)
        {
            obs_arr[i].x = 0;
            obs_arr[i].y = 0;
            obs_arr[i].num = 0;
        }
    }

    for(int i = 5; i < 320 - 5; )
    {
        float base = obs_arr[i].x;
        int flag = 0;
        if(obs_arr[i].num == 0)
        {
            i ++;
            continue;
        }
        for(int j = i - 2; j < i + 2; j++)
        {
            float cha = abs(obs_arr[j].x - base);
            if(cha < 500)   //cha < 0.5m
            {
                flag++;
            }
        }
        if(flag > 3)        //flag =  4 or 5;
        {
            obs_arr[i].num = 1;
            i += 5;
        }
        else
        {
            obs_arr[i].num = 0;
            i ++;
        }
    }

    for(int i = 0; i < 320; i++)
    {
        if(1 == obs_arr[i].num)
        {
            //        printf("obs: (%f, %f) %d %d\n", obs_arr[i].x, obs_arr[i].y, obs_arr[i].num, i);
            struct point obsN;
            obsN.x = obs_arr[i].x / 1000;
            obsN.y = obs_arr[i].y / 1000;
            obstacle->push_back(obsN);
            printf("{%f, %f},\n", obsN.x, obsN.y);
        }

    }
    //struct point obsN;
    //obsN.x = 2;
    //obsN.y = 0;
    //obstacle.push_back(obsN);
    gettimeofday(&end, NULL);
    timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    printf(" create Obs---true : %fms\n", timeuse/1000);

//    cvShowImage("y",img1f);
//    cvShowImage("consistency",img2f);
//    cvShowImage("high",img3f);

}














