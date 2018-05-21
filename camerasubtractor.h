#ifndef CAMERASUBTRACTOR_H
#define CAMERASUBTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstdlib>
#include <fstream>

void AllocateImages( const cv::Mat& I );
void accumulateBackground( cv::Mat& I );
void setHighThreshold( float scale ) ;
void setLowThreshold( float scale );
void createModelsfromStats() ;
void backgroundDiff(cv::Mat& I, cv::Mat& dst ) ;
void showForgroundInRed( char** argv, const cv::Mat &img);
void adjustThresholds(char** argv, cv::Mat &img);
void trainingImage( cv::Mat &image, int frame_count);

#endif // CAMERASUBTRACTOR_H

