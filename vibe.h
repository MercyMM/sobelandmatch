
#include <opencv2/opencv.hpp>
#include "opencv/highgui.h"
#include <iostream>

using namespace std;
using namespace cv;

#define defaultNbSamples 20		//ÿ�����ص����������
#define defaultReqMatches 2		//#minָ��
#define defaultRadius 20		//Sqthere�뾶
#define defaultSubsamplingFactor 1	//�Ӳ�������
#define background 0		//��������
#define foreground 255		//ǰ������

void Initialize(CvMat* pFrameMat,RNG rng);//��ʼ��
void update(CvMat* pFrameMat,CvMat* segMat,RNG rng,int nFrmNum,bool pixel[640*480],bool pixelnext[640*480]);//����
