#include <fstream>    
#include "opencv2/opencv.hpp"    
#include <vector>    
#include "readUbyte.h"
#include "classifier.h"

using namespace std;
using namespace cv;
   
#define NEED_STUDY	1    

int main(int argc, char *argv[])
{
	randomForest RTtree;
	SVMclassifier mySVM;
	bpNet mybpNet;

	vector<NumTrainData> trainbuffer, predictbuffer;
	int maxCount = 60000;

#if(NEED_STUDY)    
	readUbyte readertrain(string("../res/train-images.idx3-ubyte"),
		string("../res/train-labels.idx1-ubyte"));
	readertrain.ReadData(trainbuffer, maxCount);

	RTtree.train(trainbuffer);
	mySVM.train(trainbuffer);
	mybpNet.train(trainbuffer);


#else 
	RTtree.read("randomForest.xml");
	mySVM.read("SVMclassifier.xml");
	mybpNet.read("bpNet.xml");
#endif

	readUbyte readerpredict(string("../res/t10k-images.idx3-ubyte"),
		string("../res/t10k-labels.idx1-ubyte"));
	readerpredict.ReadData(predictbuffer, maxCount);

	RTtree.predict(predictbuffer);
	mySVM.predict(predictbuffer);
	mybpNet.predict(predictbuffer);

	system("pause");
	return 0;
}