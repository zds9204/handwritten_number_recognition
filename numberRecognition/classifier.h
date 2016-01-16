
#ifndef __CLASSIFIER_
#define __CLASSIFIER_

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

#include "readUbyte.h"

using namespace std;
using namespace cv;


class classifier
{
public:
	enum trainWay { HAND, AUTO };
	virtual void read(const string &XMLfilename) = 0;
	//分类器应该输入归一化的数据,不论是训练还是预测
	virtual void train(const Mat &mtrainData, const Mat &mresult, trainWay trainway = HAND) = 0;//训练分类器,默认不使用自动调参（太耗费时间）
	virtual void predict(const Mat &predictData, const Mat &mresult) = 0;	//用训练好的分类器进行检测
protected:
	void show(string classifierName,int total, int right, int error);
};

class randomForest: public classifier
{
public:
	void read(const string &XMLfilename);
	void train(const Mat &mtrainData, const Mat &mresult, trainWay trainway = HAND);
	void predict(const Mat &predictData, const Mat &mresult);
private:
	CvRTrees forest;
};

class SVMclassifier: public classifier
{
public:
	void read(const string &XMLfilename);
	void train(const Mat &mtrainData, const Mat &mresult, trainWay trainway = HAND);
	void predict(const Mat &predictData, const Mat &mresult);
private:
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
};

class bpNet : public classifier
{
public:
	void read(const string &XMLfilename);
	void train(const Mat &mtrainData, const Mat &mresult, trainWay trainway = HAND);
	void predict(const Mat &predictData, const Mat &mresult);
private:
	CvANN_MLP bp; 
	CvANN_MLP_TrainParams params;
};

#endif