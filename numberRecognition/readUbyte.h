
#ifndef __TEADUBYTE_
#define __TEADUBYTE_

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp" 

using namespace std;
using namespace cv;

#define SHOW_PROCESS	0

#define FEATURELENGTH	64
#define TRAINWIDTH		8
#define TRAINHEIGHT		8

const int featureLen = FEATURELENGTH;
const int trainwidth = TRAINWIDTH;
const int trainheight = TRAINHEIGHT;


class readUbyte
{
public:
	readUbyte(const string &imagefilename, const string &labelfilename);
	~readUbyte(){};

	void ReadData(Mat &mtrainData, Mat &mresult, int maxCount, bool IFUSEROI = true);

private:
	class NumTrainData	//用于临时存储数据的嵌套类
	{
	public:
		NumTrainData()
		{
			memset(data, 0, sizeof(data));
			result = -1;
		}
	public:
		float data[FEATURELENGTH];
		int result;
	};

private:
	ifstream lab_ifs;
	ifstream ifs;

	readUbyte(const readUbyte &name);
	void swapBuffer(char* buf);
	void GetROI(Mat& src, Mat& dst);
};

#endif