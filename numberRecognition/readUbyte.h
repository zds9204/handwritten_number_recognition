
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

class NumTrainData
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

class readUbyte
{
public:
	readUbyte(const string &imagefilename, const string &labelfilename);
	~readUbyte(){};

	void ReadData(vector<NumTrainData> &feature, int maxCount, bool IFUSEROI = true);

private:
	ifstream lab_ifs;
	ifstream ifs;
	void swapBuffer(char* buf);
	void GetROI(Mat& src, Mat& dst);

private:
	readUbyte(const readUbyte &name);
};

#endif