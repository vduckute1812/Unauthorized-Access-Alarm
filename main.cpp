//opencv
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/bgsegm.hpp>

//C
#include <stdio.h>

//C++
#include <iostream>
#include <string>
#include <sstream>

#include "camerasubtractor.h"
#include "modelSVM.h"
#include "findlinkpixel.h"
#include "rs232.h"

#include <opencv2/bgsegm.hpp>


using namespace cv;
using namespace std;

// ** Number frame for training */
#define NUM_TRAINING_FRAME 20
bool check_create_diff = false;

// Global variables
Mat frame; //current frame
Mat fgMaskMOG; //fg mask fg mask generated by MOG method
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Mat diffFrame;  //Differencing image
Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
char keyboard; //input from keyboard

//Hog detector
HOGDescriptor hog;


vector<Rect> detected, detected_filtered;


// ** Machine learning model */
cv::Ptr< cv::ml::SVM > svm;
Mat train_data;
vector< float > description;
vector<Mat> detect_lst;


///** Function Headers */
void help();
void processVideo(String videoFilename, const HOGDescriptor& hog_2);
void detect_people( Mat& frame);
string intToString(int number);
void getSubtract( Mat &img, Mat &dest);
Rect getFrameForTest(const Rect& areaDet, int max_row, int max_col);


int main(int argc, char* argv[])
{
    const char* keys =
    {
        "{help h|     | show help message}"
        "{pd    |     | path of directory contains possitive images}"
        "{nd    |     | path of directory contains negative images}"
        "{vid   |     | video for testing}"
        "{kn    |     | kernel of SVM}"
        "{c     |     | C of SVM}"
        "{g     |     | gamma of SVM}"
        "{ft    |     | train or get model}"
        "{fn    |my_detector.yml| file name of trained SVM}"
        "{b_w   |     | width block Size of HoG}"
        "{b_h   |     | height block Size of HoG}"
        "{bs_w  |     | width block stride of HoG}"
        "{bs_h  |     | height block stride of HoG}"
        "{c_w   |     | width cell size of HoG}"
        "{c_h   |     | height cell size of HoG}"
        "{n_bin |     | bin number of HoG}"
    };
    CommandLineParser parser( argc, argv, keys );

    if ( parser.has( "help" ) )
    {
        parser.printMessage();
        exit( 0 );
    }

    String obj_det_filename = parser.get< String >( "fn" );
    String vid = parser.get< String >( "vid" );
    String neg_dir = parser.get< String >( "nd" );
    String pos_dir = parser.get< String >( "pd" );
    int kernel = parser.get< int >( "kn" );
    float C = parser.get< float >( "c" );
    float gamma = parser.get< float >( "g" );    
    bool train_status = parser.get< bool >( "ft" );    
    int block_w = parser.get< int >( "b_w" );
    int block_h = parser.get< int >( "b_h" );
    int block_stride_w = parser.get< int >( "bs_w" );
    int block_stride_h = parser.get< int >( "bs_h" );
    int cell_w = parser.get< int >( "c_w" );
    int cell_h = parser.get< int >( "c_h" );
    int n_bin = parser.get< int >("n_bin");

    if( pos_dir.empty() || neg_dir.empty() )
    {
        parser.printMessage();
        cout << "Wrong number of parameters.\n\n"
             << "Example command line:\n" << argv[0] << " -dw=64 -dh=128 -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian64x128.xml -d\n"
             << "\nExample command line for testing trained detector:\n" << argv[0] << " -t -fn=HOGpedestrian64x128.xml -td=/INRIAPerson/Test/pos";
        exit( 1 );
    }

    vector< Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector< int > labels;
    clog << "Positive images are being loaded..." ;
    load_images( pos_dir, pos_lst, false );

    if ( pos_lst.size() > 0 )
    {
        clog << "...[done]" << endl;
    }
    else
    {
        clog << "no image in " << pos_dir <<endl;
        return 1;
    }
    Size pos_image_size = pos_lst[0].size();
    clog << "Negative images are being loaded...";
    load_images( neg_dir, full_neg_lst, false );
    // sample_neg( full_neg_lst, neg_lst, pos_image_size );
    clog << "...[done]" << endl;
    clog << "Histogram of Gradients are being calculated for positive images...";
    computeHOGs( pos_image_size, Size(block_w, block_h), Size(block_stride_w, block_stride_h), Size(cell_w, cell_h), n_bin, pos_lst, gradient_lst, false );
    size_t positive_count = gradient_lst.size();

    labels.assign( positive_count, +1 );
    clog << "...[done] ( positive count : " << positive_count << " )" << endl;
    clog << "Histogram of Gradients are being calculated for negative images...";
    computeHOGs( pos_image_size, Size(block_w, block_h), Size(block_stride_w, block_stride_h), Size(cell_w, cell_h), n_bin, full_neg_lst, gradient_lst, false );
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, 0 );
    CV_Assert( positive_count < labels.size() );
    clog << "...[done] ( negative count : " << negative_count << " )" << endl;
    Mat train_data;
    convert_to_ml( gradient_lst, train_data );
    clog << "Training SVM...";

    svm = trainModel( obj_det_filename, train_data, labels, kernel, gamma, C, train_status );  

    //set detector svm
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    //create GUI windows
    namedWindow("Frame");
    namedWindow("FG Mask MOG");
    // namedWindow("FG Mask MOG 2");
    // namedWindow("Differencing frame");

    //create Background Subtractor objects
    pMOG  = bgsegm::createBackgroundSubtractorMOG(); //MOG2 approach
    pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach

    //Hog extractor
    HOGDescriptor hog_2(pos_image_size,Size(block_w, block_h), Size(block_stride_w, block_stride_h), Size(cell_w, cell_h), n_bin);


    // //input data coming from a video
    processVideo(vid, hog_2);

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}

/**
* @function processVideo
*/
void processVideo(String videoFilename,  const HOGDescriptor& hog_2) {
    //create the capture object
    VideoCapture capture;
    if(isdigit(videoFilename[0])) {
         capture.open(atoi(videoFilename.c_str()));
    }
    else{
         capture.open(videoFilename);
    }

    Mat gx, gy; 
    Mat mag, angle; 

    // find link pixel
    vector< vector<point> > clusterPixel;
    pair<point,point> rectCluster;
    vector<pair<point, point> > listRect;    
    
    // find contours
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Rect boundRect;

    // frame image
    int frame_count = 0;
    int alarm_flag = 0;

    if(!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }

    //read input data. ESC or 'q' for quitting
    keyboard = 0;

    int num_port = RS232_GetPortnr("ttyUSB0");
    RS232_OpenComport(num_port, 9600, "8N1");
    bool checkAlarm = false;

    while( keyboard != 'q' && keyboard != 27 ){

        //read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

       //update the background model
        pMOG->apply(frame, fgMaskMOG);
        pMOG2->apply(frame, fgMaskMOG2);
        fgMaskMOG.convertTo(fgMaskMOG, CV_32F, 1/255.0);
        Sobel(fgMaskMOG, gx, CV_32F, 1, 0, 1);
        Sobel(fgMaskMOG, gy, CV_32F, 0, 1, 1);
        cartToPolar(gx, gy, mag, angle, 1); 

        normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);

        /// Find contours
        findContours( mag, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Draw contours
        for( int i = 0; i< contours.size(); i++ )
        {
            boundRect = boundingRect( Mat(contours[i]) );
            if(boundRect.height * boundRect.width < 1000 || boundRect.height * boundRect.width > 12000 || boundRect.width*2 >= frame.cols-1)
                continue;
            boundRect = getFrameForTest(boundRect, frame.rows, frame.cols);
            Mat frameRecog (frame, boundRect);
            resize(frameRecog, frameRecog, Size(50, 100));

            cvtColor(frameRecog, frameRecog, COLOR_BGR2GRAY);

            hog_2.compute(frameRecog, description);

            Mat data = Mat(description).clone();
            vector<Mat> detect;
            Mat data_train;
            detect.push_back(data);
            convert_to_ml(detect, data_train);
            alarm_flag+= svm->predict(data_train);

            if(frame_count%25==0){
                if(alarm_flag >= 15 && !checkAlarm){
                    cout<<"Co nguoi dot nhap trai phep"<<endl;
                    RS232_cputs(num_port, "+0967817066#");
                    checkAlarm = true;
                }
                alarm_flag=0;
            }
            // cout<<svm->predict(data_train)<<endl;   
            detect.clear();


            rectangle( fgMaskMOG, boundRect.tl(), boundRect.br(), Scalar(255), 2, 8, 0 );
        }


        if(isdigit(videoFilename[0])){
            frame_count++;
        }


        //** Differencing frame */
        // frame_count++;

        // // First frame -> Train model
        // if( frame_count <= NUM_TRAINING_FRAME ){
        //     trainingImage(frame, frame_count);
        // }
        // // Train completed
        // else{
        //     if (!check_create_diff)
        //     {
        //         // Create a differncing frame model
        //         createModelsfromStats();
        //         check_create_diff = true;
        //     }

        //     getSubtract(frame, diffFrame);

        //     Sobel(diffFrame, gx, CV_32F, 1, 0, 1);
        //     Sobel(diffFrame, gy, CV_32F, 0, 1, 1);
        //     cartToPolar(gx, gy, mag, angle, 1); 

        //     normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);

        //     /// Find contours
        //     findContours( mag, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        //     /// Draw contours
        //     for( int i = 0; i< contours.size(); i++ )
        //     {
        //         boundRect = boundingRect( Mat(contours[i]) );
        //         if(boundRect.height * boundRect.width < 1000 || boundRect.height * boundRect.width > 12000 || boundRect.width*2 >= frame.cols-1)
        //             continue;
        //         rectangle( diffFrame, boundRect.tl(), boundRect.br(), Scalar(255), 2, 8, 0 );
        //     }

        //     // show the different frame
        //     imshow("Differencing frame", diffFrame);
        // }

        if(strcmp(videoFilename.c_str(), "0") != 0){
            stringstream ss;

            ss << capture.get(CAP_PROP_POS_FRAMES);
            string frameNumberString = ss.str();
            frame_count = atoi(frameNumberString.c_str());

            //get the frame number and write it on the current frame
            rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                      cv::Scalar(255,255,255), -1);
            putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        }

        // //show the current frame and the fg masks
        // detect_people(frame);
        imshow( "Frame", frame);
        imshow( "FG Mask MOG"  , fgMaskMOG );
        // imshow( "FG Mask MOG 2", fgMaskMOG2);

        //get the input from the keyboard
        keyboard = (char)waitKey( 30 );
    }

    //delete capture object
    capture.release();
}

void detect_people( Mat& frame) {
    detected.clear();
    detected_filtered.clear();
    hog.detectMultiScale(frame, detected, 0, Size(8,8), Size(16,16), 1.06, 2);
    size_t i, j;
    /*checking for the distinctly detected human in a frame*/
    for (i=0; i<detected.size(); i++)
    {
        Rect r = detected[i];
        for (j=0; j<detected.size(); j++)
            if (j!=i && (r & detected[j]) == r)
                break;
        if (j== detected.size())
                detected_filtered.push_back(r);
    }
    /*for each distinctly detected human draw rectangle around it*/
    for (i=0; i<detected_filtered.size(); i++)
    {
        Rect r = detected_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
            rectangle(frame, r.tl(), r.br(), Scalar(0,0,255), 2);
    }
}

string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void getSubtract( Mat &img, Mat &dest){
    backgroundDiff(img, dest);
    cv::normalize(dest, dest, 0, 255, NORM_MINMAX, CV_8UC1);
}

Rect getFrameForTest(const Rect& areaDet, int max_row, int max_col){

    double ratio = (float)areaDet.height/areaDet.width;
    Rect areaRecog;

    if ( ratio < 2 ){

        int border = 2*areaDet.width - areaDet.height;
        areaRecog.x = areaDet.x;
        areaRecog.y = areaDet.y-border;

        if(areaDet.y-border < 0){
            areaRecog.y = 0;
        }


        areaRecog.width = areaDet.width;
        areaRecog.height = areaDet.width*2;
    }

    else{
        int border = int(areaDet.height-2*areaDet.width)/2;
        areaRecog.x = areaDet.x - border;
        
        if(areaDet.x - border < 0){
            areaRecog.x = 0;
        }

        areaRecog.y = areaDet.y;

        areaRecog.width = areaDet.height/2;
        areaRecog.height = areaDet.height;
    }

    // if(areaRecog.x < 0 || areaRecog.x >= max_col || areaRecog.y < 0 || areaRecog.y >= max_row
    //     || areaRecog.x+areaRecog.width >= max_col || areaRecog.y+areaDet.height >= max_row){
    //     return Rect(0,0,0,0);
    // }

    return areaRecog;
}