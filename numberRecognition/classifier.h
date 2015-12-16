
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
	virtual void train(const vector<NumTrainData>& trainData) = 0;
	virtual void predict(const vector<NumTrainData>& predictData) = 0;
protected:
	void show(string classifierName,int total, int right, int error);
};

class randomForest: public classifier
{
public:
	void train(const vector<NumTrainData>& trainData);
	void predict(const vector<NumTrainData>& predictData);
private:
	CvRTrees forest;
};

class SVMclassifier: public classifier
{
public:
	void train(const vector<NumTrainData>& trainData);
	void predict(const vector<NumTrainData>& predictData);
private:
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
};

#endif