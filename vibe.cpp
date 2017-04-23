#include <stdio.h>
// #include <tchar.h>
#include "vibe.h"
//#include <opencv2/opencv.hpp>
//#include "opencv/highgui.h"
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

static int c_xoff[9] = {-1,  -1,  -1,  1, 1, 1,  0, 0, 0};
static int c_yoff[9] = {-1,   0,   1, -1, 0, 1, -1, 0, 1};
//static int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//x���ھӵ�
//static int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//y���ھӵ�

float samples[1024][1024][defaultNbSamples+1];//����ÿ�����ص������ֵ
//int pixel[640][480];//���浱ǰͼ���ǰ���뱳��ֵ


//��ʼ��
void Initialize(CvMat* pFrameMat,RNG rng)
{

	//��¼������ɵ� ��(r) �� ��(c)
	int rand,r,c;

	//��ÿ�������������г�ʼ��
	for(int y=0;y<pFrameMat->rows;y++)//Height
	{
		for(int x=0;x<pFrameMat->cols;x++)//Width
		{
			for(int k=0;k<defaultNbSamples;k++)
			{
				//�����ȡ��������ֵ
				rand=rng.uniform( 0, 8 );
				r=y+c_yoff[rand]; if(r<0) r=0; if(r>=pFrameMat->rows) r=pFrameMat->rows-1;	//��
				c=x+c_xoff[rand]; if(c<0) c=0; if(c>=pFrameMat->cols) c=pFrameMat->cols-1;	//��
				//�洢��������ֵ
				samples[y][x][k]=CV_MAT_ELEM(*pFrameMat,float,r,c);
			}
			samples[y][x][defaultNbSamples]=0;
		}
	}
}


//���º���
void update(CvMat* pFrameMat,CvMat* segMat,RNG rng,int nFrmNum,bool pixel[],bool pixelnext[])
{

	for(int y=0;y<pFrameMat->rows;y++)	//Height
	{
		for(int x=0;x<pFrameMat->cols;x++)//Width
		{

			//�ж�һ�����Ƿ��Ǳ�����,index��¼�ѱȽϵ�����������count��ʾƥ�����������
			int count=0,index=0;float dist=0;
			//
			while((count<defaultReqMatches) && (index<defaultNbSamples))
			{
				dist=abs(CV_MAT_ELEM(*pFrameMat,float,y,x)-samples[y][x][index]);
				if(dist<defaultRadius) count++;
				index++;
			}
			if(count>=defaultReqMatches)
			{

				//�ж�Ϊ��������,ֻ�б�������ܱ����������͸��´洢����ֵ
				samples[y][x][defaultNbSamples]=0;

				*((float *)CV_MAT_ELEM_PTR(*segMat,y,x))=background;
				pixel[y*640+x]=pixelnext[y*640+x];
				pixelnext[y*640+x]=0;

				int rand=rng.uniform(0,defaultSubsamplingFactor-1);
				if(rand==0)
				{
					rand=rng.uniform(0,defaultNbSamples-1);
					samples[y][x][rand]=CV_MAT_ELEM(*pFrameMat,float,y,x);
				}
				rand=rng.uniform(0,defaultSubsamplingFactor-1);
				if(rand==0)
				{
					int xN,yN;
					rand=rng.uniform(0,8);yN=y+c_yoff[rand];if(yN<0) yN=0; if(yN>=pFrameMat->rows) yN=pFrameMat->rows-1;
					rand=rng.uniform(0,8);xN=x+c_xoff[rand];if(xN<0) xN=0; if(xN>=pFrameMat->cols) xN=pFrameMat->cols-1;
					rand=rng.uniform(0,defaultNbSamples-1);
					samples[yN][xN][rand]=CV_MAT_ELEM(*pFrameMat,float,y,x);
				} 
			}
			else 
			{
				//�ж�Ϊǰ������
				*((float *)CV_MAT_ELEM_PTR(*segMat,y,x))=foreground;
				pixel[y*640+x]=pixelnext[y*640+x];
				pixelnext[y*640+x]=1;

				samples[y][x][defaultNbSamples]++;
				if(samples[y][x][defaultNbSamples]>20)
				{
					int rand=rng.uniform(0,defaultNbSamples-1);
					if(rand==0){
						rand=rng.uniform(0,defaultNbSamples-1);
						samples[y][x][rand]=CV_MAT_ELEM(*pFrameMat,float,y,x);
					}
				}
			}

		}
	}

}
