#include "template_matching.h"

// Function to load image using OpenCV
Image loadImage(const std::string& filename) {
    cv::Mat img = cv::imread(filename);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        exit(1);
    }
    Image image;
    image.data = img;
    return image;
}

// Function to calculate Sum of Squared Differences (SSD) between two images
double calculateSSD(const cv::Mat& img1, const cv::Mat& img2) {
    // Ensure images have the same size
    assert(img1.size() == img2.size());

    double ssd = 0.0;
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            double diff = img1.at<uchar>(y, x) - img2.at<uchar>(y, x);
            ssd += diff * diff;
        }
    }
    return ssd;
}


cv::Mat ssdMatch(const cv::Mat& img, const cv::Mat& templ, const cv::Mat& orig) {
    int resultCols = img.cols - templ.cols + 1;
    int resultRows = img.rows - templ.rows + 1;
    cv::Mat output(resultRows, resultCols, CV_32S);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {

            cv::Mat roi(img, cv::Rect(j, i, templ.cols, templ.rows));
            cv::Mat diff;
            cv::absdiff(roi, templ, diff);
            cv::Mat resultImg = diff.mul(diff);
            output.at<int>(i, j) = cv::sum(resultImg)[0];
        }
    }

    cv::Point minLoc;
    cv::minMaxLoc(output, nullptr, nullptr, &minLoc);

    /**
     * Define a rectangular region of interest (ROI) within an image.
     *
     * @param x The x-coordinate of the top-left corner of the ROI.
     * @param y The y-coordinate of the top-left corner of the ROI.
     * @param width The width of the ROI.
     * @param height The height of the ROI.
     *
     * @return A cv::Rect object representing the specified ROI.
     */
    cv::Rect roi = cv::Rect(minLoc.x, minLoc.y, templ.cols, templ.rows);

    cv::rectangle(orig, roi, cv::Scalar(0, 255, 255), 2);

    return orig;
}

// Function to calculate Normalized Cross-Correlation (NCC) between two images
double calculateNCC(const cv::Mat& img1, const cv::Mat& img2) {
    // Ensure images have the same size
    assert(img1.size() == img2.size());

    double mean1 = mean(img1)[0];
    double mean2 = mean(img2)[0];

    double numerator = 0.0;
    double denominator_img1 = 0.0;
    double denominator_img2 = 0.0;

    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            double diff1 = img1.at<uchar>(y, x) - mean1;
            double diff2 = img2.at<uchar>(y, x) - mean2;
            numerator += diff1 * diff2;
            denominator_img1 += diff1 * diff1;
            denominator_img2 += diff2 * diff2;
        }
    }

    double ncc = numerator / sqrt(denominator_img1 * denominator_img2);
    return ncc;
}

// Function to calculate similarity between original image and template at a given position
double calculateSimilarity(const Image& original, const Image& templateImage, int x, int y, SimilarityMetric metric, double& elapsedTimeMicro, double& elapsedTime) {
    double similarityScore = 0.0;

    /**
    * Extracts a region of interest (ROI) from the input image.
    *
    * @param image The input image from which the ROI will be extracted.
    * @param roi A rectangle specifying the region of interest (ROI) within the image.
    * @return A new image containing the specified region of interest.
    */
    cv::Mat roi(original.data, cv::Rect(x, y, templateImage.data.cols, templateImage.data.rows));

    // Measure elapsed time for the operation in microseconds
    auto startTimeMicro = std::chrono::high_resolution_clock::now();

    similarityScore = calculateNCC(roi, templateImage.data);

    auto endTimeMicro = std::chrono::high_resolution_clock::now();
    elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTimeMicro - startTimeMicro).count();

    // Measure elapsed time for the operation in nanoseconds
    auto startTimeNano = std::chrono::high_resolution_clock::now();
    similarityScore = calculateNCC(roi, templateImage.data);
    auto endTimeNano = std::chrono::high_resolution_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeNano - startTimeNano).count();
    return similarityScore;
}

// Function to find template in original image
std::tuple<cv::Mat, double> findTemplate(const Image& original, const Image& templateImage, SimilarityMetric metric) {
    double maxSimilarityScore = std::numeric_limits<double>::min();
    int bestMatchX = 0;
    int bestMatchY = 0;
    double elapsedTimeMicro = 0.0; // Variable to store elapsed time in microseconds
    double elapsedTimeNano = 0.0; // Variable to store elapsed time in nanoseconds
    // Iterate over each pixel in the original image
    for (int y = 0; y <= original.data.rows - templateImage.data.rows; ++y) {
        for (int x = 0; x <= original.data.cols - templateImage.data.cols; ++x) {
            // Calculate similarity score between original image and template
            double similarityScore = calculateSimilarity(original, templateImage, x, y, metric, elapsedTimeMicro, elapsedTimeNano);

            // Update maximum similarity score and position
            if (similarityScore > maxSimilarityScore) {
                maxSimilarityScore = similarityScore;
                bestMatchX = x;
                bestMatchY = y;
            }
        }
    }

    std::string metricName;
    metricName = "NCC";

    std::cout << "Elapsed time (" << metricName << ") in microseconds: " << elapsedTimeMicro << std::endl;
    std::cout << "Elapsed time (" << metricName << ") in nanoseconds: " << elapsedTimeNano << std::endl;

    // Draw rectangle around matched region
    cv::rectangle(original.data, cv::Point(bestMatchX, bestMatchY),
        cv::Point(bestMatchX + templateImage.data.cols, bestMatchY + templateImage.data.rows),
        cv::Scalar(255, 0, 0), 2);

    return std::make_tuple(original.data, elapsedTimeNano);
}
