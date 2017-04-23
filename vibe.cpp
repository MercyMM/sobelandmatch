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
//static int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//x的邻居点
//static int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//y的邻居点

float samples[1024][1024][defaultNbSamples+1];//保存每个像素点的样本值
//int pixel[640][480];//保存当前图像的前景与背景值


//初始化
void Initialize(CvMat* pFrameMat,RNG rng)
{

	//记录随机生成的 行(r) 和 列(c)
	int rand,r,c;

	//对每个像素样本进行初始化
	for(int y=0;y<pFrameMat->rows;y++)//Height
	{
		for(int x=0;x<pFrameMat->cols;x++)//Width
		{
			for(int k=0;k<defaultNbSamples;k++)
			{
				//随机获取像素样本值
				rand=rng.uniform( 0, 8 );
				r=y+c_yoff[rand]; if(r<0) r=0; if(r>=pFrameMat->rows) r=pFrameMat->rows-1;	//行
				c=x+c_xoff[rand]; if(c<0) c=0; if(c>=pFrameMat->cols) c=pFrameMat->cols-1;	//列
				//存储像素样本值
				samples[y][x][k]=CV_MAT_ELEM(*pFrameMat,float,r,c);
			}
			samples[y][x][defaultNbSamples]=0;
		}
	}
}


//更新函数
void update(CvMat* pFrameMat,CvMat* segMat,RNG rng,int nFrmNum,bool pixel[],bool pixelnext[])
{

	for(int y=0;y<pFrameMat->rows;y++)	//Height
	{
		for(int x=0;x<pFrameMat->cols;x++)//Width
		{

			//判断一个点是否是背景点,index记录已比较的样本个数，count表示匹配的样本个数
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

				//判断为背景像素,只有背景点才能被用来传播和更新存储样本值
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
				//判断为前景像素
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
