#include "modelSVM.h"


using namespace cv;
using namespace cv::ml;
using namespace std;


void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;
    const int size_x = box.width;
    const int size_y = box.height;
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols >= box.width && full_neg_lst[i].rows >= box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - size_x );
            box.y = rand() % ( full_neg_lst[i].rows - size_y );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}
void computeHOGs( const cv::Size wsize, const cv::Size bsize, const cv::Size bstride, const cv::Size csize, const int n_bin, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog(wsize, bsize, bstride, csize, n_bin);
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        // if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        // {
        //     Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
        //                   ( img_lst[i].rows - wsize.height ) / 2,
        //                   wsize.width,
        //                   wsize.height);
            cvtColor( img_lst[i], gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors);
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors);
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        // }
    }
}


void test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );
    vector< String > files;
    glob( test_dir, files );
    int delay = 0;
    VideoCapture cap;
    if ( videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }
    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );
    for( size_t i=0;; i++ )
    {
        Mat img;
        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = imread( files[i] );
        }
        if ( img.empty() )
        {
            return;
        }
        vector< Rect > detections;
        vector< double > foundWeights;
        hog.detectMultiScale( img, detections, foundWeights );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
        }
        imshow( obj_det_filename, img );
        if( waitKey( delay ) == 27 )
        {
            return;
        }
    }
}

Ptr< SVM >  trainModel(String obj_det_filename, Mat train_data, vector<int> labels, int typeKernel, float gamma, float C, bool available){
    /* Default values to train SVM */
    Ptr< SVM > svm;
    if(available){
        svm = Algorithm::load<SVM>(obj_det_filename);
    }
    else{
        svm = SVM::create();
        svm->setCoef0( 0.0 );
        svm->setDegree( 3 );
        svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
        svm->setGamma( gamma );
        svm->setKernel( typeKernel );
        svm->setNu( 0.5 );
        svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
        svm->setC( C ); // From paper, soft classifier
        svm->setType( SVM::C_SVC ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
        svm->train( train_data, ROW_SAMPLE, labels );

        svm->save(obj_det_filename);        
    }

    return svm;
    // clog << "...[done]" << endl;    
}


void getFeature(const Mat& img, HOGDescriptor& hog, vector< float >& description){

    cvtColor(img, img, COLOR_BGR2GRAY);

    hog.compute(img, description);

    Mat data = Mat(description).clone();

    vector<Mat> detect;
    Mat data_train;
    detect.push_back(data);
    // detect.push_back(data_2);

    convert_to_ml(detect, data_train);
}

int svmPredict( Ptr<SVM> &svm, const Mat& data_train ){
    return svm->predict(data_train);
}
