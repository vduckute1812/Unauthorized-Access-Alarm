#include "camerasubtractor.h"


using namespace std;


// Global storage
//
// Float, 3-channel images
//
cv::Mat image;
cv::Mat IavgF, IdiffF, IprevF, IhiF, IlowF;
cv::Mat tmp, tmp2, mask;

// Float, 1-channel images
//
vector<cv::Mat> Igray(3);
vector<cv::Mat> Ilow(3);
vector<cv::Mat> Ihi(3);

// Byte, 1-channel image
//
cv::Mat Imaskt;

// Thresholds
//
float high_thresh = 30.0;  //scaling the thesholds in backgroundDiff()
float low_thresh = 38.0;

// Counts number of images learned for averaging later
//
float Icount;

// I is just a sample image for allocation purposes
// (passed in for sizing)
//
void AllocateImages( const cv::Mat& I ) {
    cv::Size sz = I.size();
    IavgF = cv::Mat::zeros(sz, CV_32FC3 );
    IdiffF = cv::Mat::zeros(sz, CV_32FC3 );
    IprevF = cv::Mat::zeros(sz, CV_32FC3 );
    IhiF = cv::Mat::zeros(sz, CV_32FC3 );
    IlowF = cv::Mat::zeros(sz, CV_32FC3 );
    Icount = 0.00001; // Protect against divide by zero
    tmp = cv::Mat::zeros( sz, CV_32FC3 );
    tmp2 = cv::Mat::zeros( sz, CV_32FC3 );
    Imaskt = cv::Mat( sz, CV_32FC1 );

}

// Learn the background statistics for one more frame
// I is a color sample of the background, 3-channel, 8u
//
void accumulateBackground(  cv::Mat& I ){
    static int first = 1; // nb. Not thread safe
    I.convertTo( tmp, CV_32F ); // convert to float
    if( !first ){
        IavgF += tmp;
        cv::absdiff( tmp, IprevF, tmp2 );
        IdiffF += tmp2;
        Icount += 1.0;
    }
    first = 0;
    IprevF = tmp;
}

void setHighThreshold( float scale ) {
    IhiF = IavgF + (IdiffF * scale);
    cv::split( IhiF, Ihi );
}

void setLowThreshold( float scale ) {
    IlowF = IavgF - (IdiffF * scale);
    cv::split( IlowF, Ilow );
}

void createModelsfromStats() {
    IavgF *= (1.0/Icount);
    IdiffF *= (1.0/Icount);

    // Make sure diff is always something
    //
    IdiffF += cv::Scalar( 1.0, 1.0, 1.0 );
    setHighThreshold( high_thresh);
    setLowThreshold( low_thresh);
}


// Create a binary: 0,255 mask where 255 (red) means foreground pixel
// I      Input image, 3-channel, 8u
// Imask  Mask image to be created, 1-channel 8u
//
void backgroundDiff (
        cv::Mat& I,
        cv::Mat& Imask) {

    I.convertTo( tmp, CV_32F ); // To float
    cv::split( tmp, Igray );

    // Channel 1
    //
    cv::inRange( Igray[0], Ilow[0], Ihi[0], Imask );

    // Channel 2
    //
    cv::inRange( Igray[1], Ilow[1], Ihi[1], Imaskt );
    Imask = cv::min( Imask, Imaskt );

    // Channel 3
    //
    cv::inRange( Igray[2], Ilow[2], Ihi[2], Imaskt );
    Imask = cv::min( Imask, Imaskt );

    // Finally, invert the results
    //
    Imask = 255 - Imask;
}


void trainingImage(cv::Mat &image, int frame_count){
    if( !image.data ) exit(1); // Something went wrong, abort
    if(frame_count==1){
        // Allocate Images
        AllocateImages(image);

    }
    accumulateBackground(image);
}

