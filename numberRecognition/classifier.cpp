
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
	cout << "==================================================================" << endl;
	cout << "| Classifier Name |" << "    Total |" << "    Right |" << "    Error |" << "    Accuracy |" << endl;
	cout << "------------------------------------------------------------------" << endl;
	cout << "|" << setw(16) << classifierName << " |" << setw(9) << total << " |"
		<< setw(9) << right << " |" << setw(9) << error << " |" << setw(12) << (double)right / total << " |" << endl;
	cout << "------------------------------------------------------------------" << endl;

	return;
}

//随机森林
void randomForest::read(const string &XMLfilename)
{
	forest.load(XMLfilename.c_str());
}

void randomForest::train(const vector<NumTrainData>& trainData, trainWay trainway)
{
	int testCount = trainData.size();

	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	cout << "Random Fores begin training ..." << endl;
	double t = (double)getTickCount();

	for (int i = 0; i< testCount; i++)
	{
		NumTrainData td = trainData.at(i);
		memcpy(data.data + i*featureLen*sizeof(float), td.data, featureLen*sizeof(float));

		res.at<unsigned int>(i, 0) = td.result;
	}

	/////////////START RT TRAINNING//////////////////    
	forest.train(data, CV_ROW_SAMPLE, res, Mat(), Mat(), Mat(), Mat(),
		CvRTParams(15, 10, 0, false, 15, 0, false, 0, 200, 0.01f, CV_TERMCRIT_ITER | CV_TERMCRIT_EPS));

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Training finished! We do this in " << t << " second." << endl;

	forest.save("randomForest.xml");
}

void randomForest::predict(const vector<NumTrainData>& predictData)
{
	Mat sample = Mat::zeros(1, featureLen, CV_32FC1);
	int predictResult, label;
	int total = 0, right = 0, error = 0;

	cout << "Random Fores begin predicting ..." << endl;
	double t = (double)getTickCount();

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

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Predicting finished! We do this in " << t << " second." << endl;

	return;
}


//SVM
void SVMclassifier::read(const string &XMLfilename)
{
	svm.load(XMLfilename.c_str());
}


void SVMclassifier::train(const vector<NumTrainData>& trainData, trainWay trainway)
{
	int testCount = trainData.size();

	Mat m = Mat::zeros(1, featureLen, CV_32FC1);
	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	cout << "SVM begin training ..." << endl;
	double t = (double)getTickCount();

	for (int i = 0; i< testCount; i++)
	{
		NumTrainData td = trainData.at(i);
		memcpy(m.data, td.data, featureLen*sizeof(float));
		normalize(m, m);	//SVM需要对数据进行归一化
		memcpy(data.data + i*featureLen*sizeof(float), m.data, featureLen*sizeof(float));

		res.at<unsigned int>(i, 0) = td.result;
	}

	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.5, 1.0, 62.0, 0.5, 0.1, NULL, criteria);

	if (trainway == AUTO)
	{
		svm.train_auto(data, res, Mat(), Mat(), param, 10);
		CvSVMParams params = svm.get_params();
		cout << "SVM parameters:" << endl;
		cout << "    C" << " coef0" << " degree" << "   gamma" << " kernel" << "  p" << "   cweights" <<endl;
		cout << setw(5) << params.C << setw(6) << params.coef0 << setw(7) << params.degree << setprecision(4) << setw(8)
			<< params.gamma << setw(7) << params.kernel_type << setw(3) << params.nu << setw(3) << params.p;
		(params.class_weights == NULL) ? setw(10) : setw(params.class_weights->cols);
		cout << params.class_weights << endl;
	}
	else
		svm.train(data, res, Mat(), Mat(), param);

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Training finished! We do this in " << t << " second." << endl;

	svm.save("SVMclassifier.xml");
}

void SVMclassifier::predict(const vector<NumTrainData>& predictData)
{
	Mat sample = Mat::zeros(1, featureLen, CV_32FC1);
	int predictResult, label;
	int total = 0, right = 0, error = 0;

	cout << "SVM begin predicting ..." << endl;
	double t = (double)getTickCount();

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

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Predicting finished! We do this in " << t << " second." << endl;

	return;
}

