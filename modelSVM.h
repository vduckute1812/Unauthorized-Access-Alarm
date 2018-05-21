#ifndef MODELSVM_H
#define MODELSVM_H

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <time.h>

void convert_to_ml( const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );
void load_images( const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages );
void sample_neg( const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size );
void computeHOGs( const cv::Size wsize, const cv::Size bsize, const cv::Size bstride, const cv::Size csize, const int n_bin, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip );
void test_trained_detector( cv::String obj_det_filename, cv::String test_dir, cv::String videofilename );
cv::Ptr< cv::ml::SVM >  trainModel(cv::String obj_det_filename, cv::Mat train_data, std::vector<int> labels, int typeKernel, float gamma, float C, bool available=false);
void getFeature(const cv::Mat& img, cv::HOGDescriptor& hog, std::vector< float >& description);
int svmPredict( cv::Ptr<cv::ml::SVM> &svm, const cv::Mat& data_train );

#endif