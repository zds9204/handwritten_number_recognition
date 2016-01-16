#include <fstream>    
#include "opencv2/opencv.hpp"    
#include <vector>    
#include "readUbyte.h"
#include "classifier.h"
#include "dimReduction.h"

using namespace std;
using namespace cv;
   
#define NEED_STUDY	1    

int main(int argc, char *argv[])
{
	randomForest RTtree;
	SVMclassifier mySVM;
	bpNet mybpNet;

	Mat trainMat, predictMat;
	Mat normTrainMat, normPredictMat, trainlabel, predictlabel;
	int maxCount = 1000;

  
	readUbyte readertrain(string("../res/train-images.idx3-ubyte"),
		string("../res/train-labels.idx1-ubyte"));
	readertrain.ReadData(trainMat, trainlabel, maxCount);

	/*onlynormalize normalize;
	normalize.Init(trainMat);
	normalize.protect(trainMat, normTrainMat);*/
	PCAdimReduction pca;
	pca.Init(trainMat, 0.95);
	pca.protect(trainMat, normTrainMat);

#if(NEED_STUDY)  
	RTtree.train(normTrainMat, trainlabel);
	mySVM.train(normTrainMat, trainlabel);
	mybpNet.train(normTrainMat, trainlabel);
#else 
	RTtree.read("randomForest.xml");
	mySVM.read("SVMclassifier.xml");
	mybpNet.read("bpNet.xml");
#endif

	readUbyte readerpredict(string("../res/t10k-images.idx3-ubyte"),
		string("../res/t10k-labels.idx1-ubyte"));
	readerpredict.ReadData(predictMat, predictlabel, maxCount);


	/*normalize.protect(predictMat, normPredictMat);*/
	pca.protect(predictMat, normPredictMat);

	RTtree.predict(normPredictMat, predictlabel);
	mySVM.predict(normPredictMat, predictlabel);
	mybpNet.predict(normPredictMat, predictlabel);

	system("pause");
	return 0;
}