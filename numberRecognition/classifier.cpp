
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "readUbyte.h"
#include "classifier.h"

using namespace std;
using namespace cv;

void classifier::show(string classifierName, int total, int right, int error)
{
	cout << "================================================================" << endl;
	cout << "| Classifier Name |" << "    Total|" << "    Right|" << "    Error|" << "    Accuracy|" << endl;
	cout << "----------------------------------------------------------------" << endl;
	cout << "|" << setw(17) << classifierName << "|" << setw(9) << total << "|"
		<< setw(9) << right << "|" << setw(9) << error << "|" << setw(12) << (double)right / total << "|" << endl;
	cout << "----------------------------------------------------------------" << endl;

	return;
}

//Ëæ»úÉ­ÁÖ
void randomForest::train(const vector<NumTrainData>& trainData)
{
	int testCount = trainData.size();

	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	for (int i = 0; i< testCount; i++)
	{
		NumTrainData td = trainData.at(i);
		memcpy(data.data + i*featureLen*sizeof(float), td.data, featureLen*sizeof(float));

		res.at<unsigned int>(i, 0) = td.result;
	}

	/////////////START RT TRAINNING//////////////////    
	forest.train(data, CV_ROW_SAMPLE, res, Mat(), Mat(), Mat(), Mat(),
		CvRTParams(10, 10, 0, false, 15, 0, true, 4, 100, 0.01f, CV_TERMCRIT_ITER));
	forest.save("randomForest.xml");
}

void randomForest::predict(const vector<NumTrainData>& predictData)
{
	Mat sample = Mat::zeros(1, featureLen, CV_32FC1);
	int predictResult, label;
	int total = 0, right = 0, error = 0;

	for (vector<NumTrainData>::const_iterator iter = predictData.begin();
		iter != predictData.end(); iter ++)
	{
		for (int i = 0; i < featureLen; i++)
			sample.at<float>(0, i) = (*iter).data[i];

		label = (*iter).result;
		predictResult = cvRound(forest.predict(sample));

		if (predictResult == label)
			right++;
		else
			error++;

		total++;
	}

	show("randomForest", total, right, error);

	return;
}


//SVM
void SVMclassifier::train(const vector<NumTrainData>& trainData)
{
	int testCount = trainData.size();

	Mat m = Mat::zeros(1, featureLen, CV_32FC1);
	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	for (int i = 0; i< testCount; i++)
	{
		NumTrainData td = trainData.at(i);
		memcpy(m.data, td.data, featureLen*sizeof(float));
		normalize(m, m);
		memcpy(data.data + i*featureLen*sizeof(float), m.data, featureLen*sizeof(float));

		res.at<unsigned int>(i, 0) = td.result;
	}

	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);

	svm.train(data, res, Mat(), Mat(), param);
	svm.save("SVMclassifier.xml");
}

void SVMclassifier::predict(const vector<NumTrainData>& predictData)
{
	Mat sample = Mat::zeros(1, featureLen, CV_32FC1);
	int predictResult, label;
	int total = 0, right = 0, error = 0;

	for (vector<NumTrainData>::const_iterator iter = predictData.begin();
		iter != predictData.end(); iter++)
	{
		for (int i = 0; i < featureLen; i++)
			sample.at<float>(0, i) = (*iter).data[i];

		label = (*iter).result;
		normalize(sample, sample);
		predictResult = cvRound(svm.predict(sample));

		if (predictResult == label)
			right++;
		else
			error++;

		total++;
	}

	show("SVM", total, right, error);

	return;
}

