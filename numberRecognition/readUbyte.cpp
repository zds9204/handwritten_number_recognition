
#include <fstream>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp" 
#include "readUbyte.h"

using namespace std;
using namespace cv;

readUbyte::readUbyte(const string &imagefilename, const string &labelfilename)
{
	lab_ifs.open(labelfilename, ios_base::binary);
	ifs.open(imagefilename, ios_base::binary);

	try
	{
		if (ifs.fail() == true)
		{
			string error = "Open file: " + imagefilename + " failed!";
			throw runtime_error(error);
		}
		if (lab_ifs.fail() == true)
		{
			string error = "Open file: " + labelfilename + " failed!";
			throw runtime_error(error);
		}
	}
	catch (const runtime_error &e)
	{
		cerr << e.what() << endl;
		exit(-1);
	}
}

void readUbyte::ReadData(Mat &mtrainData, Mat &mresult, int maxCount, bool IFUSEROI)
{
	vector<NumTrainData> trainData;

	char magicNum[4], ccount[4], crows[4], ccols[4];
	ifs.read(magicNum, sizeof(magicNum));
	ifs.read(ccount, sizeof(ccount));
	ifs.read(crows, sizeof(crows));
	ifs.read(ccols, sizeof(ccols));

	int count, rows, cols;
	swapBuffer(ccount);
	swapBuffer(crows);
	swapBuffer(ccols);

	memcpy(&count, ccount, sizeof(count));
	memcpy(&rows, crows, sizeof(rows));
	memcpy(&cols, ccols, sizeof(cols));

	//Just skip label header    
	lab_ifs.read(magicNum, sizeof(magicNum));
	lab_ifs.read(ccount, sizeof(ccount));

	//Create source and show image matrix    
	Mat src = Mat::zeros(rows, cols, CV_8UC1);
	Mat temp = Mat::zeros(trainheight, trainwidth, CV_8UC1);
	Mat img, dst;

	char label = 0;
	Scalar templateColor(255, 0, 255);

	NumTrainData rtd;
  
	int total = 0;
	while (!ifs.eof())
	{
		if (total >= count)
			break;

		total++;
		//cout << total << endl;

		//Read label    
		lab_ifs.read(&label, 1);
		label = label + '0';

		//Read source data    
		ifs.read((char*)src.data, rows * cols);

		if (IFUSEROI == true)
			GetROI(src, dst);
		
#if(SHOW_PROCESS)    
		//Too small to watch    
		img = Mat::zeros(dst.rows * 10, dst.cols * 10, CV_8UC1);
		resize(dst, img, img.size());

		stringstream ss;
		ss << "Number " << label;
		string text = ss.str();
		putText(img, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.0, templateColor);

		imshow("img", img);

		if (waitKey(0) == 27) //ESC to quit    
			break;
#endif    

		rtd.result = label;
		resize(dst, temp, temp.size());
		//threshold(temp, temp, 10, 1, CV_THRESH_BINARY);    

		for (int i = 0; i < trainwidth; i++)
		{
			const uchar *inData = temp.ptr<uchar>(i);
			for (int j = 0; j < trainheight; j++)
			{
				rtd.data[i * trainwidth + j] = inData[j];
			}
		}

		trainData.push_back(rtd);

		maxCount--;
		if (maxCount == 0)
			break;
	}

	ifs.close();
	lab_ifs.close();

	//将结果转换为矩阵的形式存储
	mtrainData.create(static_cast<int>(trainData.size()), featureLen, CV_32FC1);
	mresult.create(static_cast<int>(trainData.size()), 1, CV_32SC1);
	for (unsigned int i = 0; i < trainData.size(); i++)
	{
		float *ptrainData = mtrainData.ptr<float>(i);
		int *presult = mresult.ptr<int>(i);

		for (unsigned int j = 0; j < featureLen; j++)
			ptrainData[j] = trainData[i].data[j];

		presult[0] = trainData[i].result;
	}

	return ;
}



void readUbyte::swapBuffer(char* buf)
{
	char temp;
	temp = *(buf);
	*buf = *(buf + 3);
	*(buf + 3) = temp;

	temp = *(buf + 1);
	*(buf + 1) = *(buf + 2);
	*(buf + 2) = temp;
}

void readUbyte::GetROI(Mat& src, Mat& dst)
{
	int left, right, top, bottom;
	left = src.cols;
	right = 0;
	top = src.rows;
	bottom = 0;

	//Get valid area    
	for (int i = 0; i < src.rows; i++)
	{
		const uchar *inData = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (inData[j] > 0)
			{
				if (j<left) left = j;
				if (j>right) right = j;
				if (i<top) top = i;
				if (i>bottom) bottom = i;
			}
		}
	}
   

	int width = right - left;
	int height = bottom - top;
	int len = (width < height) ? height : width;

	//Create a squre    
	dst = Mat::zeros(len, len, CV_8UC1);

	//Copy valid data to squre center    
	Rect dstRect((len - width) / 2, (len - height) / 2, width, height);
	Rect srcRect(left, top, width, height);
	Mat dstROI = dst(dstRect);
	Mat srcROI = src(srcRect);
	srcROI.copyTo(dstROI);
}