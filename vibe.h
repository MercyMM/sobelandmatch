
#include <opencv2/opencv.hpp>
#include "opencv/highgui.h"
#include <iostream>

using namespace std;
using namespace cv;

#define defaultNbSamples 20		//每个像素点的样本个数
#define defaultReqMatches 2		//#min指数
#define defaultRadius 20		//Sqthere半径
#define defaultSubsamplingFactor 1	//子采样概率
#define background 0		//背景像素
#define foreground 255		//前景像素

void Initialize(CvMat* pFrameMat,RNG rng);//初始化
void update(CvMat* pFrameMat,CvMat* segMat,RNG rng,int nFrmNum,bool pixel[640*480],bool pixelnext[640*480]);//更新
