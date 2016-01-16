
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "opencv2/opencv.hpp"

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

void randomForest::train(const Mat &mtrainData, const Mat &mresult, trainWay trainway)
{
	cout << "random forest begin training ..." << endl;
	double t = (double)getTickCount();
	/////////////START RT TRAINNING//////////////////    
	forest.train(mtrainData, CV_ROW_SAMPLE, mresult, Mat(), Mat(), Mat(), Mat(),
		CvRTParams(15, 10, 0, false, 15, 0, false, 0, 200, 0.01f, CV_TERMCRIT_ITER | CV_TERMCRIT_EPS));

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Training finished! We do this in " << t << " second." << endl;

	forest.save("randomForest.xml");
}

void randomForest::predict(const Mat &predictData, const Mat &mresult)
{
	int predictResult;
	int total = 0, right = 0, error = 0;

	cout << "Random Fores begin predicting ..." << endl;
	double t = (double)getTickCount();

	for (unsigned int i = 0; i != predictData.rows; i++)
	{
		const Mat predictData_i = predictData.row(i);

		predictResult = cvRound(forest.predict(predictData_i));

		if (predictResult == mresult.at<int>(i, 0))
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


void SVMclassifier::train(const Mat &mtrainData, const Mat &mresult, trainWay trainway)
{
	cout << "SVM begin training ..." << endl;
	double t = (double)getTickCount();

	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.5, 1.0, 62.0, 0.5, 0.1, NULL, criteria);

	if (trainway == AUTO)
	{
		svm.train_auto(mtrainData, mresult, Mat(), Mat(), param, 10);
		CvSVMParams params = svm.get_params();
		cout << "SVM parameters:" << endl;
		cout << "    C" << " coef0" << " degree" << "   gamma" << " kernel" << "  p" << "   cweights" <<endl;
		cout << setw(5) << params.C << setw(6) << params.coef0 << setw(7) << params.degree << setprecision(4) << setw(8)
			<< params.gamma << setw(7) << params.kernel_type << setw(3) << params.nu << setw(3) << params.p;
		(params.class_weights == NULL) ? setw(10) : setw(params.class_weights->cols);
		cout << params.class_weights << endl;
	}
	else
		svm.train(mtrainData, mresult, Mat(), Mat(), param);

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Training finished! We do this in " << t << " second." << endl;

	svm.save("SVMclassifier.xml");
}

void SVMclassifier::predict(const Mat &predictData, const Mat &mresult)
{
	int predictResult;
	int total = 0, right = 0, error = 0;

	cout << "SVM begin predicting ..." << endl;
	double t = (double)getTickCount();

	for (unsigned int i = 0; i != predictData.rows; i++)
	{
		const Mat predictData_i = predictData.row(i);

		predictResult = cvRound(svm.predict(predictData_i));
		if (predictResult == mresult.at<int>(i, 0))
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


//bp神经网络
void bpNet::read(const string &XMLfilename)
{
	bp.load(XMLfilename.c_str());
}

void bpNet::train(const Mat &mtrainData, const Mat &mresult, trainWay trainway)
{
	cout << "bpNet begin training ..." << endl;
	double t = (double)getTickCount();

	Mat res = Mat::zeros(mresult.rows, 10, CV_32FC1);
	for (int i = 0; i < mresult.rows; i++)
		res.at<float>(i, (mresult.at<int>(i, 0) - '0')) = (float)1;


	/////////////START bpNet TRAINNING//////////////////
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;
	CvTermCriteria TermCrlt;
	TermCrlt.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	TermCrlt.epsilon = 0.01f;
	TermCrlt.max_iter = 5000;
	params.term_crit = TermCrlt;

	Mat layerSizes = (Mat_<int>(1, 3) << mtrainData.cols, 100, 10);

	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);

	bp.train(mtrainData, res, Mat(), Mat(), params);

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Training finished! We do this in " << t << " second." << endl;

	bp.save("bpNet.xml");
}

void bpNet::predict(const Mat &predictData, const Mat &mresult)
{
	Mat predictResult = Mat::zeros(1, 10, CV_32FC1);
	int total = 0, right = 0, error = 0;

	cout << "bpNet begin predicting ..." << endl;
	double t = (double)getTickCount();
	int result = 0;
	double max = 0.0;
	Point point;

	for (unsigned int i = 0; i != predictData.rows; i++)
	{
		bp.predict(predictData.row(i), predictResult);

		minMaxLoc(predictResult, NULL, &max, NULL, &point);	//找到最大值
		result = point.x * cvRound(max);

		if ((result + '0') == mresult.at<int>(i, 0))
			right ++;
		else
			error ++;

		total ++;
	}

	show("bpNet", total, right, error);

	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Predicting finished! We do this in " << t << " second." << endl;

	return;
}