

#ifndef __DIMREDUCTION_
#define __DIMREDUCTION_

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


class dimReduction
{
public:
	virtual void Init(const Mat &trainData, double retainedVariance = 0.95, bool readfromfile = true) = 0;	//降维工具初始化。第三个参数标志降维工具具体信息从存储文件中获得
	virtual void protect(const Mat &data, Mat &result) = 0;													//将一组数据投影到主成分空间中
	virtual void backProtect(const Mat &data, Mat &result) = 0;												//将主成分空间中的数据
protected:
	void showMessage(const string method, int inputLength, int outputLength);								//显示降维信息
	Mat toGrayscale(InputArray _src);
};

class onlynormalize : public dimReduction
{
public:
	void Init(const Mat &trainData, double retainedVariance = 0, bool readfromfile = 0);
	void protect(const Mat &data, Mat &result);
	void backProtect(const Mat &data, Mat &result);
};

class PCAdimReduction : public dimReduction
{
public:
	void Init(const Mat &trainData, double retainedVariance = 0.95, bool readfromfile = 0);
	void protect(const Mat &data, Mat &result);
	void backProtect(const Mat &data, Mat &result);
private:
	cv::PCA pca;
	static const string PCA_MEAN;
	static const string PCA_EIGEN_VECTOR;
};

#endif