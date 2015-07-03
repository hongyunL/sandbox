#include <iostream>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <dirent.h>
#include <fstream>

using namespace cv;
using namespace std;

//////////////////////// INPUT PARAMETERS START ////////////////////////

float sigmac            = 16;   // The sigma values that are used in computing the color weights

// (WARNING!! Division operations are implemented with bitshift operations for efficiency. Please set "histogramSize1D" to a power of two)

int histogramSize1D     = 8;    // Number of histogram bins per channel


// You can use either a video file or a image directory as input using "dataPath" variable

string dataPath;    // The directory of the input images or the path of the input video
string fileFormat;  // The file format of the input images
string savePath;    // The directory of the output images


//////////////////////// INPUT PARAMETERS END ////////////////////////

int histogramSize2D     = histogramSize1D * histogramSize1D;
int histogramSize3D     = histogramSize2D * histogramSize1D;
int logSize             = (int)log2(histogramSize1D);
int logSize2            = 2*logSize;

Mat squares             = Mat::zeros(1, 10000, CV_32FC1);

float* squaresPtr       = squares.ptr<float>(0);

vector<Mat> LAB;

vector<float> L, A, B;

float meanVectorFloat[4] = {0.5555,    0.6449,    0.0002,    0.0063};

float inverseCovarianceFloat[4][4] = {{43.3777,    1.7633,   -0.4059,    1.0997},
    {1.7633,   40.7221,   -0.0165,    0.0447},
    {-0.4059,   -0.0165,   87.0455,   -3.2744},
    {1.0997,    0.0447,   -3.2744,  125.1503}};

Mat modelMean                   = Mat(4, 1, CV_32FC1, meanVectorFloat);
Mat modelInverseCovariance      = Mat(4, 4, CV_32FC1, inverseCovarianceFloat);

void readFiles(vector<string>& imagePaths,
               vector<string>& imageNames,
               VideoCapture &cap,
               bool isimage){
    
    if(!isimage){
        cap        = VideoCapture(dataPath.c_str());
        
        if(!cap.isOpened())
            return;
    
        for (int i = 0; i < cap.get(CV_CAP_PROP_FRAME_COUNT); i++) {
            string temp = to_string(i) + ".jpg";
            imageNames.push_back(temp);
        }
    }
    else{
        DIR *imDir = opendir(dataPath.c_str());
        for( dirent *imp = readdir(imDir); imp != NULL; imp = readdir(imDir) ) {
            string imName      = imp->d_name;
            bool check = imName.size() > 3 && imName.substr( imName.size()-3 ) == fileFormat;
            if(imName.size() > 3 && check){
                imagePaths.push_back( dataPath + imName );
                imageNames.push_back( imName );
            }
        }
    }

}

void calculateHistogram(Mat im,
                        Mat &averageX,
                        Mat &averageY,
                        Mat &averageX2,
                        Mat &averageY2,
                        vector<float> &LL,
                        vector<float> &AA,
                        vector<float> &BB,
                        Mat &histogram,
                        Mat &histogramIndex){
    
    Mat lab, Lshift, Ashift, Bshift;
    
    double minL, maxL, minA, maxA, minB, maxB;
    
    averageX        = Mat::zeros(1, histogramSize3D, CV_32FC1);
    averageY        = Mat::zeros(1, histogramSize3D, CV_32FC1);
    averageX2       = Mat::zeros(1, histogramSize3D, CV_32FC1);
    averageY2       = Mat::zeros(1, histogramSize3D, CV_32FC1);
    
    // Instead scaling LAB channels, we use compute shift values to stretch the LAB histogram
    
    cvtColor(im, lab, CV_BGR2Lab);
    
    split(lab, LAB);
    
    minMaxLoc(LAB[0], &minL, &maxL);
    minMaxLoc(LAB[1], &minA, &maxA);
    minMaxLoc(LAB[2], &minB, &maxB);
    
    float tempL = (255 - maxL + minL) / (maxL - minL + 1e-3);
    float tempA = (255 - maxA + minA) / (maxA - minA + 1e-3);
    float tempB = (255 - maxB + minB) / (maxB - minB + 1e-3);
    
    Lshift          = Mat::zeros(1, 256, CV_32SC1);
    Ashift          = Mat::zeros(1, 256, CV_32SC1);
    Bshift          = Mat::zeros(1, 256, CV_32SC1);
    
    for (int i = 0; i < 256; i++) {
        
        Lshift.at<int>(0,i) = tempL * (i - minL) - minL;
        Ashift.at<int>(0,i) = tempA * (i - minA) - minA;
        Bshift.at<int>(0,i) = tempB * (i - minB) - minB;
        
    }
    
    // Calculate quantized LAB values
    
    minL = minL / 2.56;
    maxL = maxL / 2.56;
    
    minA = minA - 128;
    maxA = maxA - 128;
    
    minB = minB - 128;
    maxB = maxB - 128;
    
    tempL   = float(maxL - minL)/histogramSize1D;
    tempA   = float(maxA - minA)/histogramSize1D;
    tempB   = float(maxB - minB)/histogramSize1D;
    
    float sL = float(maxL - minL)/histogramSize1D/2 + minL;
    float sA = float(maxA - minA)/histogramSize1D/2 + minA;
    float sB = float(maxB - minB)/histogramSize1D/2 + minB;
    
    for (int i = 0; i < histogramSize3D; i++) {
        
        int lpos = i % histogramSize1D;
        int apos = i % histogramSize2D / histogramSize1D;
        int bpos = i / histogramSize2D;
        
        LL.push_back(lpos * tempL + sL);
        AA.push_back(apos * tempA + sA);
        BB.push_back(bpos * tempB + sB);
        
    }
    
    // Calculate LAB histogram
    
    histogramIndex          = Mat::zeros(im.rows, im.cols, CV_32SC1);
    histogram               = Mat::zeros(1, histogramSize3D, CV_32SC1);
    
    int*    histogramPtr    = histogram.ptr<int>(0);
    
    float* averageXPtr      = averageX.ptr<float>(0);
    float* averageYPtr      = averageY.ptr<float>(0);
    float* averageX2Ptr     = averageX2.ptr<float>(0);
    float* averageY2Ptr     = averageY2.ptr<float>(0);
    
    int*    LshiftPtr       = Lshift.ptr<int>(0);
    int*    AshiftPtr       = Ashift.ptr<int>(0);
    int*    BshiftPtr       = Bshift.ptr<int>(0);
    
    int histShift = 8 - logSize;
    
    for (int y = 0; y < im.rows; y++) {
        
        int*    histogramIndexPtr   = histogramIndex.ptr<int>(y);
        
        uchar*    LPtr   = LAB[0].ptr<uchar>(y);
        uchar*    APtr   = LAB[1].ptr<uchar>(y);
        uchar*    BPtr   = LAB[2].ptr<uchar>(y);
        
        for (int x = 0; x < im.cols; x++) {
            
            // Instead of division, we use bit-shift operations for efficieny. This is valid if number of bins is a power of two (4, 8, 16 ...)
            
            int lpos                = (LPtr[x] + LshiftPtr[LPtr[x]]) >> histShift;
            int apos                = (APtr[x] + AshiftPtr[APtr[x]]) >> histShift;
            int bpos                = (BPtr[x] + BshiftPtr[BPtr[x]]) >> histShift;
            
            int index               = lpos + (apos << logSize) + (bpos << logSize2);
            
            histogramIndexPtr[x]    = index;
            
            histogramPtr[index]++;
            
            // These values are collected here for efficiency. They will later be used in computing the spatial center and variances of the colors
            
            averageXPtr[index]      += x;
            averageYPtr[index]      += y;
            averageX2Ptr[index]     += squaresPtr[x];
            averageY2Ptr[index]     += squaresPtr[y];
            
        }
    }
    
}

int precomputeParameters(Mat histogram,
                          vector<float> LL,
                          vector<float> AA,
                          vector<float> BB,
                          int numberOfPixels,
                          vector<int> &reverseMap,
                          Mat &map,
                          Mat &colorDistance,
                          Mat &exponentialColorDistance){
    
    int*    histogramPtr    = histogram.ptr<int>(0);

    Mat problematic         = Mat::zeros(histogram.cols, 1, CV_32SC1);
    
    Mat closestElement      = Mat::zeros(histogram.cols, 1, CV_32SC1);
    
    Mat sortedHistogramIdx;
    
    // The number of colors are further reduced here. A threshold is calculated so that we take the colors that can represent 95% of the image.
    
    sortIdx(histogram, sortedHistogramIdx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
    
    int* sortedHistogramIdxPtr = sortedHistogramIdx.ptr<int>(0);
    
    float energy = 0;
    
    int binCountThreshold = 0;
    
    float energyThreshold = 0.95 * numberOfPixels;
    
    for (int i = 0; i < histogram.cols; i++) {
        
        energy += (float)histogramPtr[sortedHistogramIdxPtr[i]];
        
        if (energy > energyThreshold){
            
            binCountThreshold = histogramPtr[sortedHistogramIdx.at<int>(0,i)];
            
            break;
            
        }
    }
    
    // Calculate problematic histogram bins (i.e. bins that have very few or no pixels)
    
    for (int i = 0; i < histogram.cols; i++)
        if (histogramPtr[i] < binCountThreshold)
            problematic.at<int>(i,0) = 1;
    
    map = Mat::zeros(1, histogram.cols, CV_32SC1);
    
    int* mapPtr = map.ptr<int>(0);
    
    int count = 0;
    
    for (int i = 0; i < histogram.cols; i++) {
        
        if (histogramPtr[i] >= binCountThreshold) {
            
            // Save valid colors for later use.
            
            L.push_back(LL[i]);
            A.push_back(AA[i]);
            B.push_back(BB[i]);
            
            mapPtr[i] = count;
            
            reverseMap.push_back(i);
            
            count++;
        }
        else if(histogramPtr[i] < binCountThreshold && histogramPtr[i] > 0){
            
            float mini = 1e6;
            
            int closest = 0;
            
            // Calculate the perceptually closest color of bins with a few pixels.
            
            for (int k = 0; k < histogram.cols; k++) {
                
                // Don't forget to check this, we don't want to assign them to empty histogram bins.
                
                if (!problematic.at<int>(k,0)){
                    
                    float dd = pow((LL[i] - LL[k]),2) + pow((AA[i] - AA[k]),2) + pow((BB[i] - BB[k]),2);
                    
                    if (dd < mini) {
                        mini = dd;
                        closest = k;
                    }
                }
                
            }
            
            closestElement.at<int>(i,0) = closest;
            
        }
        
    }
    
    for (int i = 0; i < histogram.cols; i++)
        if(problematic.at<int>(i,0))
            mapPtr[i] = mapPtr[closestElement.at<int>(i,0)];
    
    int numberOfColors = (int)L.size();
    
    // Precompute the color weights here
    
    exponentialColorDistance    = Mat::zeros(numberOfColors, numberOfColors, CV_32FC1);
    
    colorDistance               = Mat::zeros(numberOfColors, numberOfColors, CV_32FC1);
    
    for (int i = 0; i < numberOfColors; i++) {
        
        colorDistance.at<float>(i,i)            = 0;
        
        exponentialColorDistance.at<float>(i,i) = 1.0;
        
        for (int k = i + 1; k < numberOfColors; k++) {
            
            float colorDifference                   = pow(L[i] - L[k],2) + pow(A[i] - A[k],2) + pow(B[i] - B[k],2);
            
            colorDistance.at<float>(i,k)            = sqrt(colorDifference);
            
            colorDistance.at<float>(k,i)            = sqrt(colorDifference);
            
            exponentialColorDistance.at<float>(i,k) = exp(- colorDifference / (2 * sigmac * sigmac));
            
            exponentialColorDistance.at<float>(k,i) = exponentialColorDistance.at<float>(i,k);
            
        }
    }
    
    return numberOfColors;
    
}

void bilateralFiltering(Mat colorDistance,
                        Mat exponentialColorDistance,
                        vector<int> reverseMap,
                        int* histogramPtr,
                        float* averageXPtr,
                        float* averageYPtr,
                        float* averageX2Ptr,
                        float* averageY2Ptr,
                        Mat &mx,
                        Mat &my,
                        Mat &Vx,
                        Mat &Vy,
                        Mat &contrast){
    
    int numberOfColors = colorDistance.cols;
    
    Mat X       = Mat::zeros(1, numberOfColors, CV_32FC1);
    Mat Y       = Mat::zeros(1, numberOfColors, CV_32FC1);
    Mat X2      = Mat::zeros(1, numberOfColors, CV_32FC1);
    Mat Y2      = Mat::zeros(1, numberOfColors, CV_32FC1);
    Mat NF      = Mat::zeros(1, numberOfColors, CV_32FC1);
    
    float* XPtr     = X.ptr<float>(0);
    float* YPtr     = Y.ptr<float>(0);
    float* X2Ptr    = X2.ptr<float>(0);
    float* Y2Ptr    = Y2.ptr<float>(0);
    float* NFPtr    = NF.ptr<float>(0);
    
    // Here, we calculate the color contrast and the necessary parameters to compute the spatial center and variances

    contrast    = Mat::zeros(1, numberOfColors, CV_32FC1);
    
    float* contrastPtr  = contrast.ptr<float>(0);
    
    for (int i = 0; i < numberOfColors; i++) {
        
        float* colorDistancePtr             = colorDistance.ptr<float>(i);
        float* exponentialColorDistancePtr  = exponentialColorDistance.ptr<float>(i);
        
        for (int k = 0; k < numberOfColors; k++) {

            contrastPtr[i]  += colorDistancePtr[k] * histogramPtr[reverseMap[k]];
            
            XPtr[i]         += exponentialColorDistancePtr[k] * averageXPtr[reverseMap[k]];
            YPtr[i]         += exponentialColorDistancePtr[k] * averageYPtr[reverseMap[k]];
            X2Ptr[i]        += exponentialColorDistancePtr[k] * averageX2Ptr[reverseMap[k]];
            Y2Ptr[i]        += exponentialColorDistancePtr[k] * averageY2Ptr[reverseMap[k]];
            NFPtr[i]        += exponentialColorDistancePtr[k] * histogramPtr[reverseMap[k]];

        }
    }
    
    divide(X,   NF, X);
    divide(Y,   NF, Y);
    divide(X2,  NF, X2);
    divide(Y2,  NF, Y2);
    
    // The mx, my, Vx, and Vy represent the same symbols in the paper. They are the spatial center and variances of the colors, respectively.

    X.assignTo(mx);
    Y.assignTo(my);
    
    Vx = X2 - mx.mul(mx);
    Vy = Y2 - my.mul(my);

}

void calculateProbability(Mat mx,
                          Mat my,
                          Mat Vx,
                          Mat Vy,
                          Mat modelMean,
                          Mat modelInverseCovariance,
                          int width,
                          int height,
                          Mat &Xsize,
                          Mat &Ysize,
                          Mat &Xcenter,
                          Mat &Ycenter,
                          Mat &shapeProbability){
    
    // Convert the spatial center and variances to vector "g" in the paper, so we can compute the probability of saliency.
    
    sqrt(12 * Vx, Xsize);
    Xsize       = Xsize/(float)width;
    
    sqrt(12 * Vy, Ysize);
    Ysize       = Ysize/(float)height;
    
    Xcenter = (mx - width /2)/(float)width;
    Ycenter = (my - height/2)/(float)height;
    
    Mat     g;
    
    vconcat(Xsize, Ysize, g);
    vconcat(g, Xcenter, g);
    vconcat(g, Ycenter, g);
    
    Mat repeatedMeanVector;
    
    repeat(modelMean, 1, Xcenter.cols, repeatedMeanVector);
    
    g = g - repeatedMeanVector;
    
    g = g/2;
    
    shapeProbability = Mat::zeros(1, Xcenter.cols, CV_32FC1);
    
    float* shapeProbabilityPtr = shapeProbability.ptr<float>(0);
    
    // Comptuing the probability of saliency. As we will perform a normalization later, there is no need to multiply it with a constant term of the Gaussian function.
    
    for (int i = 0; i < Xcenter.cols; i++) {
        
        Mat result, transposed;
        
        transpose(g.col(i), transposed);
        
        gemm(transposed, modelInverseCovariance, 1.0, 0.0, 0.0, result);
        
        gemm(result, g.col(i), 1.0, 0.0, 0.0, result);
        
        shapeProbabilityPtr[i] = exp(- result.at<float>(0,0) / 2);
        
    }
    
}

void computeSaliencyMap(Mat shapeProbability,
                        Mat contrast,
                        Mat exponentialColorDistance,
                        Mat histogramIndex,
                        int* mapPtr,
                        Mat& SM,
                        Mat& saliency){
    
    double minVal, maxVal;
    
    int numberOfColors  = shapeProbability.cols;
    
    saliency            = shapeProbability.mul(contrast);
    
    float* saliencyPtr  = saliency.ptr<float>(0);
    
    for (int i = 0; i < numberOfColors; i++) {
        
        float a1 = 0;
        float a2 = 0;
        
        for (int k = 0; k < numberOfColors; k++) {
            
            if (exponentialColorDistance.at<float>(i,k) > 0.0){
                
                a1 += saliencyPtr[k] * exponentialColorDistance.at<float>(i,k);
                a2 += exponentialColorDistance.at<float>(i,k);
                
            }
            
        }
        
        saliencyPtr[i] = a1/a2;
    }

    minMaxLoc(saliency, &minVal, &maxVal);
    
    saliency = saliency - minVal;
    saliency = 255 * saliency / (maxVal - minVal) + 1e-3;
    
    minMaxLoc(saliency, &minVal, &maxVal);

    for (int y = 0; y < SM.rows; y++){
        
        uchar* SMPtr = SM.ptr<uchar>(y);
        
        int* histogramIndexPtr = histogramIndex.ptr<int>(y);
        
        for (int x = 0; x < SM.cols; x++){
            
            float sal = saliencyPtr[mapPtr[histogramIndexPtr[x]]];

            SMPtr[x] = (uchar)(sal);
            
        }
    }

    
}

void outputHowToUse(){
    
    cout << "FASA: Fast, Accurate, and Size-Aware Salient Object Detection" << endl;
    cout << "-------------------------------------------------------------" << endl;
    cout << "How to Use? There are 2 ways!" << endl;
    cout << "FASA -i -p /path/to/image/folder/ -f image_format -s /path/to/output/folder/" << endl;
    cout << "FASA -v -p /path/to/video/file.avi -s /path/to/output/folder/" << endl;
    
}

int main(int argc, const char * argv[]){
    
    bool isimage;
    
    if(argc < 2){
        
        outputHowToUse();
        return 0;
        
    }
    else{
        if(!strcmp(argv[1], "-i")){
            isimage = true;
            if(argc != 8){
                outputHowToUse();
                return 0;
            }
            else{
                for(int i = 2; i < 8; i += 2){
                    
                    const char* input = argv[i];
                    
                    if(!strcmp(input,"-p")){
                        dataPath = argv[i + 1];
                    }
                    else if(!strcmp(input,"-f")){
                        fileFormat = argv[i + 1];
                    }
                    else if(!strcmp(input,"-s")){
                        savePath = argv[i + 1];
                    }
                    else{
                        cout << "Invalid input type " << input << endl;
                        outputHowToUse();
                        return 0;
                    }
                }
            }
        }
        else if(!strcmp(argv[1], "-v")){
            isimage = false;
            if(argc != 6){
                outputHowToUse();
                return 0;
            }
            else{
                for(int i = 2; i < 6; i += 2){
                    
                    const char* input = argv[i];
                    
                    if(!strcmp(input,"-p")){
                        dataPath = argv[i + 1];
                    }
                    else if(!strcmp(input,"-s")){
                        savePath = argv[i + 1];
                    }
                    else{
                        cout << "Invalid input type " << input << endl;
                        outputHowToUse();
                        return 0;
                    }
                }
            }
        }
        else{
            outputHowToUse();
            return 0;
        }
    }
    
    if (dataPath.back() != '/') {
        dataPath = dataPath + "/";
    }
    if (savePath.back() != '/') {
        savePath = savePath + "/";
    }
    
    clock_t st, et;
    
    vector<string> imagePaths;
    vector<string> imageNames;
    
    VideoCapture cap;
    
    readFiles(imagePaths, imageNames, cap, isimage);
    
    if(!isimage && !cap.isOpened())
        return 0;
    
    if(isimage && imagePaths.size() == 0){
        cout << "No images with format " << fileFormat << " were found under " << dataPath << endl;
        return 0;
    }
    
    if(isimage)
        cout << imagePaths.size() << " image(s) were found." << endl;
    else
        cout << imageNames.size() << " video frame(s) were found." << endl;

    cout << "Processing..." << endl;

    system(("mkdir -p " + savePath).c_str());
    system(("mkdir -p " + savePath + "globalContrast/").c_str());
    system(("mkdir -p " + savePath + "saliencyProbability/").c_str());
    system(("mkdir -p " + savePath + "saliencyMaps/").c_str());
    system(("mkdir -p " + savePath + "ellipseDetection/").c_str());
    system(("mkdir -p " + savePath + "rectangleDetection/").c_str());
    system(("mkdir -p " + savePath + "rectangleBoundingBoxes/").c_str());
    
    int totalImages     = 0;
    
    float totalColor    = 0;
    
    float totalPixels   = 0;
    
    float totalTime     = 0;
    
    for (int i = 0; i < squares.cols; i++)
        squaresPtr[i] = pow(i,2);

    while(1) {
        
        Mat im;
        
        if(isimage)
            im = imread(imagePaths[totalImages]);
        else{
            cap >> im;
        }
        
        if(im.data){

            Mat lab;
            
            totalPixels += im.cols*im.rows;

            cout << "//////////////////////// SALIENCY COMPUTATION STARTS HERE ////////////////////////\n";
            
//////////////////////// SALIENCY COMPUTATION STARTS HERE ////////////////////////

            st = clock();
            
            LAB.clear();
            L.clear();
            A.clear();
            B.clear();
            
            Mat averageX, averageY, averageX2, averageY2, histogram, histogramIndex;
            
            vector<float> LL, AA, BB;
            
            calculateHistogram(im,
                               averageX,
                               averageY,
                               averageX2,
                               averageY2,
                               LL,
                               AA,
                               BB,
                               histogram,
                               histogramIndex);
            
            float* averageXPtr      = averageX.ptr<float>(0);
            float* averageYPtr      = averageY.ptr<float>(0);
            float* averageX2Ptr     = averageX2.ptr<float>(0);
            float* averageY2Ptr     = averageY2.ptr<float>(0);
            
            int*    histogramPtr    = histogram.ptr<int>(0);
            
            Mat map, colorDistance, exponentialColorDistance;
            
            vector<int> reverseMap;
            
            int numberOfColors = precomputeParameters(histogram,
                                                      LL,
                                                      AA,
                                                      BB,
                                                      im.cols * im.rows,
                                                      reverseMap,
                                                      map,
                                                      colorDistance,
                                                      exponentialColorDistance);
            
            totalColor += numberOfColors;
            
            int* mapPtr = map.ptr<int>(0);
            
            Mat mx, my, Vx, Vy, contrast;
            
            bilateralFiltering(colorDistance,
                               exponentialColorDistance,
                               reverseMap,
                               histogramPtr,
                               averageXPtr,
                               averageYPtr,
                               averageX2Ptr,
                               averageY2Ptr,
                               mx,
                               my,
                               Vx,
                               Vy,
                               contrast);
            
            Mat Xsize, Ysize, Xcenter, Ycenter, shapeProbability;
            
            calculateProbability(mx,
                                 my,
                                 Vx,
                                 Vy,
                                 modelMean,
                                 modelInverseCovariance,
                                 im.cols,
                                 im.rows,
                                 Xsize,
                                 Ysize,
                                 Xcenter,
                                 Ycenter,
                                 shapeProbability);
            
            
            Mat SM = Mat::zeros(im.rows, im.cols, CV_8UC1);
            
            Mat saliency;
            
            computeSaliencyMap(shapeProbability,
                               contrast,
                               exponentialColorDistance,
                               histogramIndex,
                               mapPtr,
                               SM,
                               saliency);
            
            et = clock();
            
            cout << "//////////////////////// SALIENCY COMPUTATION ENDS HERE ////////////////////////\n";
//////////////////////// SALIENCY COMPUTATION ENDS HERE ////////////////////////
            
            totalTime += double(et-st)/CLOCKS_PER_SEC;
            
            float* saliencyPtr  = saliency.ptr<float>(0);
            
            Mat ellipseDetection, rectangleDetection;
            
            im.copyTo(ellipseDetection);
            im.copyTo(rectangleDetection);
            
            ofstream objectRectangles;

            objectRectangles.open(savePath + "rectangleBoundingBoxes/" + imageNames[totalImages].substr(0,imageNames[totalImages].length()-4) + "txt");
            
            for (int i = 0; i < numberOfColors; i++) {
                
                float rx = Xsize.at<float>(0,i)*im.cols;
                float ry = Ysize.at<float>(0,i)*im.rows;
                
                float xx,yy,ww,hh;
                
                xx = mx.at<float>(0,i) - rx/2 >= 0 ? mx.at<float>(0,i) - rx/2 : 0;
                yy = my.at<float>(0,i) - ry/2 >= 0 ? my.at<float>(0,i) - ry/2 : 0;
                ww = xx + rx < im.cols ? rx : im.cols - xx;
                hh = yy + ry < im.rows ? ry : im.rows - yy;
                
                objectRectangles << xx << "," << yy << "," << ww << "," << hh << "," << saliencyPtr[i] << "\n";
                
                if (saliencyPtr[i] > 254){
                    ellipse(ellipseDetection, Point(mx.at<float>(0,i),my.at<float>(0,i)), Size(rx/2,ry/2),0, 0, 360, Scalar(0,0,saliencyPtr[i]), 3, CV_AA);
                    rectangle(rectangleDetection, Point(xx,yy), Point(xx+ww,yy+hh), Scalar(0,0,255), 3, CV_AA);
                }

            }
            
            objectRectangles.close();

            double minVal, maxVal;
            
            minMaxLoc(shapeProbability, &minVal, &maxVal);
            
            shapeProbability = shapeProbability - minVal;
            shapeProbability = shapeProbability / (maxVal - minVal + 1e-3);

            minMaxLoc(contrast, &minVal, &maxVal);
            
            contrast = contrast - minVal;
            contrast = contrast / (maxVal - minVal + 1e-3);
            
            float* shapeProbabilityPtr      = shapeProbability.ptr<float>(0);
            float* contrastPtr              = contrast.ptr<float>(0);
            
            Mat saliencyProbabilityImage    = Mat::zeros(im.rows, im.cols, CV_32FC1);
            Mat globalContrastImage         = Mat::zeros(im.rows, im.cols, CV_32FC1);
            
            for (int y = 0; y < im.rows; y++){
                
                float* saliencyProbabilityImagePtr  = saliencyProbabilityImage.ptr<float>(y);
                float* globalContrastImagePtr       = globalContrastImage.ptr<float>(y);
                
                int* histogramIndexPtr = histogramIndex.ptr<int>(y);
                
                for (int x = 0; x < im.cols; x++){
                    
                    saliencyProbabilityImagePtr[x]  = shapeProbabilityPtr[mapPtr[histogramIndexPtr[x]]];
                    globalContrastImagePtr[x]       = contrastPtr[mapPtr[histogramIndexPtr[x]]];
                    
                }
            }
            
            saliencyProbabilityImage = 255 * saliencyProbabilityImage;
            saliencyProbabilityImage.convertTo(saliencyProbabilityImage, CV_8UC1);
            
            globalContrastImage = 255 * globalContrastImage;
            globalContrastImage.convertTo(globalContrastImage, CV_8UC1);
            
            imwrite(savePath + "globalContrast/"        + imageNames[totalImages].substr(0,imageNames[totalImages].size()-3) + "png", globalContrastImage);
            imwrite(savePath + "saliencyProbability/"   + imageNames[totalImages].substr(0,imageNames[totalImages].size()-3) + "png", saliencyProbabilityImage);
            imwrite(savePath + "saliencyMaps/"          + imageNames[totalImages].substr(0,imageNames[totalImages].size()-3) + "png", SM);
            
            imwrite(savePath + "ellipseDetection/"      + imageNames[totalImages].substr(0,imageNames[totalImages].size()-3) + "png", ellipseDetection);
            imwrite(savePath + "rectangleDetection/"    + imageNames[totalImages].substr(0,imageNames[totalImages].size()-3) + "png", rectangleDetection);
            

//            cout << "error in number of images\n";
            if(totalImages % (imageNames.size()/10) == 0){
                float percentage = ((float) 100 * totalImages) / imageNames.size();
                cout << percentage << "%" << endl;
            }
            
            totalImages++;
            
        }
        else
            break;
        
    }
    
    cout << "Number of images: " << totalImages << endl;
    cout << "Average processing time: " << totalTime / totalImages * 1e3 << " ms" << endl;
    cout << "Processing speed: " << totalImages / totalTime << " image/second" << endl;
    cout << "Average number of colors: " << totalColor / totalImages << endl;
    
    return 0;
}


