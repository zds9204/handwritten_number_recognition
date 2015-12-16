#include <fstream>    
#include "opencv2/opencv.hpp"    
#include <vector>    
#include "readUbyte.h"
#include "classifier.h"

using namespace std;
using namespace cv;

//#define SHOW_PROCESS 1    
#define ON_STUDY 1    

int main(int argc, char *argv[])
{
#if(ON_STUDY)    
	vector<NumTrainData> trainbuffer, predictbuffer;
	int maxCount = 60000;
	readUbyte readertrain(string("../res/train-images.idx3-ubyte"),
		string("../res/train-labels.idx1-ubyte"));
	readertrain.ReadData(trainbuffer, maxCount);

	readUbyte readerpredict(string("../res/t10k-images.idx3-ubyte"),
		string("../res/t10k-labels.idx1-ubyte"));
	readerpredict.ReadData(predictbuffer, maxCount);


	randomForest RTtree;
	RTtree.train(trainbuffer);
	RTtree.predict(predictbuffer);

	SVMclassifier mySVM;
	mySVM.train(trainbuffer);
	mySVM.predict(predictbuffer);

#else    
	//newRtPredict();    
	//newSvmPredict();
#endif   

	system("pause");
	return 0;
}