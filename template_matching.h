#ifndef TEMPLATE_MATCHING_H
#define TEMPLATE_MATCHING_H
// Define Image structure to represent grayscale images
#include "opencv2/core/mat.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

struct Image {
    cv::Mat data;
};

// Enum for similarity metric
typedef enum class SimilarityMetric {
    SSD,
    NCC
};



// Function to load image using OpenCV
Image loadImage(const std::string& filename);

// Function to calculate Sum of Squared Differences (SSD) between two images
double calculateSSD(const cv::Mat& img1, const cv::Mat& img2);

cv::Mat ssdMatch(const cv::Mat& img, const cv::Mat& templ, const cv::Mat& orig);
// Function to calculate Normalized Cross-Correlation (NCC) between two images
double calculateNCC(const cv::Mat& img1, const cv::Mat& img2);

// Function to calculate similarity between original image and template at a given position
double calculateSimilarity(const Image& original, const Image& templateImage, int x, int y, SimilarityMetric metric, double& elapsedTimeMicro, double& elapsedTime);

// Function to find template in original image
std::tuple<cv::Mat, double> findTemplate(const Image& original, const Image& templateImage, SimilarityMetric metric);

#endif // TEMPLATE_MATCHING_H
