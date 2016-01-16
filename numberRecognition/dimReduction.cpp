
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "readUbyte.h"
#include "dimReduction.h"

using namespace std;
using namespace cv;


Mat dimReduction::toGrayscale(InputArray _src) 
{
	Mat src = _src.getMat();
	// only allow one channel
	if (src.channels() != 1) {
		CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
	}
	// create and return normalized image
	Mat dst;
	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

void dimReduction::showMessage(const string method, int inputLength, int outputLength)
{
	cout << "==================================================================" << endl;
	cout << " Dimension reduction method:" << setw(37) << method << endl;
	cout << " Length of input vector:" << setw(41) << inputLength << endl;
	cout << " Length of output vector:" << setw(40) << outputLength << endl;
	cout << "------------------------------------------------------------------" << endl;
}


//不进行降维，仅对数据进行归一化
void onlynormalize::Init(const Mat &trainData, double retainedVariance, bool readfromfile)
{
	if (trainData.cols != featureLen)
	{
		cout << "onlynormalize 输入数据长度不符合要求！" << endl;
		exit(-1);
	}

	return;
}

void onlynormalize::protect(const Mat &alldata, Mat &result)
{
	result = alldata.clone();
	Mat result_i;
	for (unsigned int i = 0; i < alldata.rows; i++)
	{
		result_i = result.row(i);							//矩阵与大矩阵共用一块数据区域
		normalize(result_i, result_i);						//归一化
	}

	showMessage("onlynormalize", featureLen, featureLen);	//输出降维结果信息

	return;
}

void onlynormalize::backProtect(const Mat &data, Mat &result)
{
	if (data.cols != featureLen)
	{
		cout << "onlynormalize 反向投影输入数据长度不符合要求！" << endl;
		exit(-1);
	}

	result = toGrayscale(data);

	return;
}

//PCA降维                          
const string PCAdimReduction::PCA_MEAN = "mean";
const string PCAdimReduction::PCA_EIGEN_VECTOR = "eigen_vector";

void PCAdimReduction::Init(const Mat &trainData, double retainedVariance, bool readfromfile)
{
	if (readfromfile == true)
	{
		FileStorage fs_r("PCA.xml", FileStorage::READ);
		fs_r[PCA_MEAN] >> pca.mean;
		fs_r[PCA_EIGEN_VECTOR] >> pca.eigenvectors;
		fs_r.release();
	}
	else
	{
		if (trainData.cols != featureLen)
		{
			cout << "PCA输入数据长度不符合要求！" << endl;
			exit(-1);
		}

		pca.computeVar(trainData, cv::Mat(), CV_PCA_DATA_AS_ROW, retainedVariance);

		FileStorage fs_w("PCA.xml", FileStorage::WRITE); 
		fs_w << PCA_MEAN << pca.mean;
		fs_w << PCA_EIGEN_VECTOR << pca.eigenvectors;
		fs_w.release();
	}

	showMessage("PCA", featureLen, pca.eigenvalues.rows);			//输出降维结果信息

	return;
}

//输入全部数据构成的矩阵，得到降维并归一化的矩阵
void PCAdimReduction::protect(const Mat &alldata, Mat &result)
{
	if (alldata.cols != featureLen)
	{
		cout << "PCA计算输入数据长度不符合要求！" << endl;
		exit(-1);
	}

	result.create(alldata.rows, pca.eigenvalues.rows, CV_32FC1);
	Mat result_i;
	for (unsigned int i = 0; i < alldata.rows; i++)
	{
		result_i = result.row(i);					//矩阵与大矩阵共用一块数据区域
		pca.project(alldata.row(i), result_i);		//PCA降维
		normalize(result_i, result_i);				//归一化
	}

	return;
}

void PCAdimReduction::backProtect(const Mat &data, Mat &result)
{
	if (data.cols != pca.eigenvalues.cols)
	{
		cout << "PCA反向投影输入数据长度不符合要求！" << endl;
		exit(-1);
	}

	pca.backProject(data, result);
	result = toGrayscale(result);

	return;
}
