#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "sift.h"
#include "template_matching.h"
using namespace cv;
using namespace std;
using namespace std::chrono; // Namespace for timing

/**
 * Apply Gaussian blur to the input image.
 *
 * @param input The input image to be blurred.
 * @param sigma The standard deviation of the Gaussian kernel.
 * @return The blurred image.
 */
Mat applyGaussianBlur(const Mat& input, double sigma) {
    // Ensure sigma is non-negative
    if (sigma < 0) {
        std::cerr << "Error: Sigma value must be non-negative." << std::endl;
        return Mat(); // Return an empty matrix if sigma is negative
    }

    // Calculate kernel size based on sigma
    int kernel_size = cvRound(6 * sigma) + 1; // The kernel size increases with the sigma value

    // Create an empty kernel matrix
    Mat kernel(kernel_size, kernel_size, CV_64F);

    // Calculate constants
    double mean = kernel_size / 2; // Mean value for symmetric kernel
    double scale = 1.0 / (sigma * sqrt(2 * CV_PI)); // Scaling factor for Gaussian function

    // Fill the kernel values using Gaussian function
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            double x = i - mean; // Distance from center along x-axis
            double y = j - mean; // Distance from center along y-axis
            double value = scale * exp(-(x * x + y * y) / (2 * sigma * sigma)); // Gaussian function
            kernel.at<double>(i, j) = value; // Assign the computed value to the kernel
        }
    }

    // Normalize the kernel to ensure sum of elements equals 1
    normalize(kernel, kernel, 1, 0, NORM_L1);


    // Apply the convolution operation using the calculated kernel
    Mat blurred;
    filter2D(input, blurred, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

    return blurred; // Return the blurred image
}



/**
 * Filter keypoints based on response threshold.
 *
 * @param keypoints Vector of keypoints to be filtered.
 * @param image The image associated with the keypoints.
 * @param response_threshold The minimum response value required for a keypoint to be retained.
 */
void filterKeypoints(std::vector<KeyPoint>& keypoints, const Mat& image, double response_threshold) {
    std::vector<KeyPoint> filtered_keypoints;
    for (const auto& kp : keypoints) {
        // Check if the keypoint's response value is above the threshold
        if (kp.response >= response_threshold) {
            filtered_keypoints.push_back(kp); // Add the keypoint to the filtered list
        }
    }
    keypoints = filtered_keypoints; // Update the keypoints with the filtered list
}


/**
 * Resize the input image by a specified scale factor using bilinear interpolation.
 *
 * @param input The input image to be resized.
 * @param output The resized output image.
 * @param scale_factor The scaling factor for resizing.
 */
void resizeImage(const Mat& input, Mat& output, double scale_factor) {
    // Check if the input image is empty
    if (input.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Calculate new size based on the scale factor
    int new_width = cvRound(input.cols * scale_factor);
    int new_height = cvRound(input.rows * scale_factor);

    // Create output image with new size
    output.create(new_height, new_width, input.type());

    // Loop through each pixel in the output image
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            // Map the coordinates in the output image to the coordinates in the input image
            float src_x = x / scale_factor;
            float src_y = y / scale_factor;

            // Get the four neighboring pixels in the input image
            int x1 = cvFloor(src_x);
            int y1 = cvFloor(src_y);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // Ensure that the coordinates are within the input image boundaries
            x1 = std::max(0, std::min(x1, input.cols - 1));
            x2 = std::max(0, std::min(x2, input.cols - 1));
            y1 = std::max(0, std::min(y1, input.rows - 1));
            y2 = std::max(0, std::min(y2, input.rows - 1));

            // Calculate the interpolation weights
            float dx2 = src_x - x1;
            float dy2 = src_y - y1;
            float dx1 = 1.0 - dx2;
            float dy1 = 1.0 - dy2;

            // Perform bilinear interpolation
            output.at<Vec3b>(y, x) =
                input.at<Vec3b>(y1, x1) * dx1 * dy1 +
                input.at<Vec3b>(y1, x2) * dx2 * dy1 +
                input.at<Vec3b>(y2, x1) * dx1 * dy2 +
                input.at<Vec3b>(y2, x2) * dx2 * dy2;
        }
    }
}

/**
 * Build an image pyramid with specified number of levels.
 *
 * @param image The input image to build the pyramid from.
 * @param pyramid Vector to store the image pyramid levels.
 * @param levels Number of levels in the pyramid.
 * @param scale_factor Scaling factor for resizing each level.
 */
void buildImagePyramid(const Mat& image, std::vector<Mat>& pyramid, int levels, double scale_factor) {
    // Check if the input image is empty
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return; // Return if the input image is empty
    }

    // Clear the pyramid vector
    pyramid.clear();
    // Add the original image to the pyramid
    pyramid.push_back(image.clone());

    // Resize and add subsequent levels to the pyramid
    Mat current_level = image.clone();
    for (int i = 1; i < levels; ++i) {
        // Resize the current level using the specified scale factor
        Size new_size(cvRound(current_level.cols * scale_factor), cvRound(current_level.rows * scale_factor));
        Mat next_level(new_size, current_level.type());
        resizeImage(current_level, next_level, scale_factor); // Resize the current level
        pyramid.push_back(next_level); // Add the resized image to the pyramid
        current_level = next_level.clone(); // Update the current level for the next iteration
    }
}

/**
 * Detect keypoints in each level of the image pyramid using ORB detector.
 *
 * @param pyramid The image pyramid containing different levels of the image.
 * @param keypoints Vector to store the detected keypoints.
 * @param response_threshold The minimum response value required for a keypoint to be retained.
 */
void detectKeypointsDOG(const std::vector<Mat>& pyramid, std::vector<KeyPoint>& keypoints, double response_threshold) {
    // ORB (Oriented FAST and Rotated BRIEF)
    // Ptr<FeatureDetector> : for managing memory allocation and deallocation automatically.
    Ptr<FeatureDetector> detector = ORB::create(); // Create ORB detector

    for (int i = 0; i < pyramid.size(); ++i) {
        std::vector<KeyPoint> kps;
        detector->detect(pyramid[i], kps); // Detect keypoints using ORB
        filterKeypoints(kps, pyramid[i], response_threshold); // Filter keypoints based on response threshold
        keypoints.insert(keypoints.end(), kps.begin(), kps.end()); // Add detected keypoints to the vector
    }
}


/**
 * Convert Cartesian gradients to polar coordinates.
 *
 * @param grad_x Gradient in the x-direction.
 * @param grad_y Gradient in the y-direction.
 * @param mag Magnitude of gradients.
 * @param angle Angle of gradients.
 * @param angleInDegrees Flag indicating whether the angle should be in degrees or radians.
 */
void carttoPolar(const Mat& grad_x, const Mat& grad_y, Mat& mag, Mat& angle, bool angleInDegrees) {
    // Ensure the input matrices have the same size
    if (grad_x.size() != grad_y.size()) {
        std::cerr << "Error: Gradient matrices must have the same size." << std::endl;
        return; // Return if the gradient matrices have different sizes
    }

    // Initialize output matrices
    mag.create(grad_x.size(), CV_32F); // Magnitude matrix
    angle.create(grad_x.size(), CV_32F); // Angle matrix

    // Loop through each pixel in the input matrices
    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            // Compute magnitude
            float dx = grad_x.at<float>(y, x);
            float dy = grad_y.at<float>(y, x);
            float magnitude = sqrt(dx * dx + dy * dy); // Compute magnitude of gradient
            mag.at<float>(y, x) = magnitude; // Assign magnitude to the output matrix

            // Compute angle
            float angle_radians = atan2(dy, dx); // Compute angle in radians
            if (angleInDegrees) {
                // Convert angle to degrees if required
                angle.at<float>(y, x) = angle_radians * 180.0 / CV_PI;
            }
            else {
                angle.at<float>(y, x) = angle_radians; // Assign angle in radians to the output matrix
            }
        }
    }
}

/**
 * Compute descriptors for keypoints using SIFT.
 *
 * @param image The input image.
 * @param keypoints Vector of keypoints detected in the image.
 * @param descriptors Matrix to store computed descriptors.
 */
void computeDescriptors(const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& descriptors) {
    // SIFT descriptor parameters
    const int descriptor_size = 128; // SIFT descriptor size
    const int patch_size = 16; // Size of the patch around each keypoint
    const int half_patch = patch_size / 2; // Half size of the patch

    // Compute gradients using Sobel operators
    Mat grad_x, grad_y;
    Sobel(image, grad_x, CV_32F, 1, 0); // Compute gradient in x-direction
    Sobel(image, grad_y, CV_32F, 0, 1); // Compute gradient in y-direction

    // Compute magnitude and orientation of gradients
    Mat mag, angle;
    cartToPolar(grad_x, grad_y, mag, angle, true); // Convert gradients to polar coordinates

    // Iterate over keypoints
    descriptors.create(keypoints.size(), descriptor_size, CV_32F); // Initialize descriptor matrix
    for (size_t i = 0; i < keypoints.size(); ++i) {
        // Get coordinates of keypoint
        int x = keypoints[i].pt.x;
        int y = keypoints[i].pt.y;

        // Extract patch around keypoint
        Mat patch;
        image(Rect(x - half_patch, y - half_patch, patch_size, patch_size)).copyTo(patch);

        // Compute histogram of gradients in the patch
        Mat hist(1, descriptor_size, CV_32F, Scalar(0));
        for (int r = 0; r < patch.rows; ++r) {
            for (int c = 0; c < patch.cols; ++c) {
                float patch_angle = angle.at<float>(y - half_patch + r, x - half_patch + c);
                int bin = int(patch_angle / 45); // Divide 360 degrees into 8 bins
                hist.at<float>(0, bin) += mag.at<float>(y - half_patch + r, x - half_patch + c); // Accumulate gradient magnitude
            }
        }

        // Normalize histogram
        normalize(hist, hist, 1, 0, NORM_L2);

        // Store histogram as descriptor
        hist.copyTo(descriptors.row(i));
    }
}

/**
 * Compute Hamming distance between two binary descriptors.
 *
 * @param desc1 First binary descriptor.
 * @param desc2 Second binary descriptor.
 * @return Hamming distance between the two descriptors.
 */
int hammingDistance(const Mat& desc1, const Mat& desc2) {
    // Compute Hamming distance between two binary descriptors
    int distance = 0;
    for (int i = 0; i < desc1.cols; ++i) {
        distance += cv::norm(desc1.col(i) != desc2.col(i), NORM_L1);
    }
    return distance;
}


/**
 * Draw matches between keypoints of two images.
 *
 * @param image1 The first input image.
 * @param keypoints1 Keypoints detected in the first image.
 * @param image2 The second input image.
 * @param keypoints2 Keypoints detected in the second image.
 * @param matches Matches between keypoints of the two images.
 * @param output Output image to draw the matches.
 */
void drawMatchesFeatures(const Mat& image1, const std::vector<KeyPoint>& keypoints1,
                         const Mat& image2, const std::vector<KeyPoint>& keypoints2,
                         const std::vector<DMatch>& matches, Mat& output) {
    // Merge images horizontally
    int max_height = std::max(image1.rows, image2.rows);
    int total_width = image1.cols + image2.cols;
    output.create(max_height, total_width, CV_8UC3); // Create output image
    output.setTo(Scalar(255, 255, 255)); // Fill background with white

    // Draw images
    Mat left_roi(output, Rect(0, 0, image1.cols, image1.rows));
    image1.copyTo(left_roi); // Copy first image to left region of output image
    Mat right_roi(output, Rect(image1.cols, 0, image2.cols, image2.rows));
    image2.copyTo(right_roi); // Copy second image to right region of output image

    // Draw matches with randomly chosen colors
    RNG rng; // Random number generator
    for (size_t i = 0; i < matches.size(); ++i) {
        const KeyPoint& kp1 = keypoints1[matches[i].queryIdx]; // Keypoint in first image
        const KeyPoint& kp2 = keypoints2[matches[i].trainIdx]; // Keypoint in second image

        // Offset keypoints for second image
        Point2f kp2_offset(kp2.pt.x + image1.cols, kp2.pt.y);

        // Generate random color for drawing
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // Draw line between matched keypoints with random color
        line(output, kp1.pt, kp2_offset, color, 1);

        // Draw circles around keypoints with random color
        circle(output, kp1.pt, 5, color, 2); // Circle around keypoint in first image
        circle(output, kp2_offset, 5, color, 2); // Circle around keypoint in second image
    }
}

/**
 * Compute Sum of Squared Differences (SSD) between two descriptors.
 *
 * @param desc1 First descriptor.
 * @param desc2 Second descriptor.
 * @return SSD between the two descriptors.
 */
double computeSSD(const Mat& desc1, const Mat& desc2) {
    // Ensure descriptor sizes are the same
    if (desc1.size() != desc2.size()) {
        std::cerr << "Error: Descriptor sizes are not the same." << std::endl;
        return -1;
    }

    // Compute SSD
    double ssd = 0.0;
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            double diff = desc1.at<double>(i, j) - desc2.at<double>(i, j);
            ssd += diff * diff; // Accumulate squared differences
        }
    }
    return ssd;
}

/**
 * Compute Normalized Cross-Correlation (NCC) between two descriptors.
 *
 * @param desc1 First descriptor.
 * @param desc2 Second descriptor.
 * @return NCC between the two descriptors.
 */
double computeNCC(const Mat& desc1, const Mat& desc2) {
    // Ensure descriptor sizes are the same
    if (desc1.size() != desc2.size()) {
        std::cerr << "Error: Descriptor sizes are not the same." << std::endl;
        return -1; // Return -1 if descriptor sizes are different
    }

    // Compute mean and standard deviation of each descriptor
    double mean1 = 0.0, mean2 = 0.0;
    double sum_sq_diff1 = 0.0, sum_sq_diff2 = 0.0, sum_product = 0.0;
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            mean1 += desc1.at<double>(i, j);
            mean2 += desc2.at<double>(i, j);
        }
    }
    mean1 /= (desc1.rows * desc1.cols); // Compute mean of first descriptor
    mean2 /= (desc2.rows * desc2.cols); // Compute mean of second descriptor

    // Compute correlation
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            double diff1 = desc1.at<double>(i, j) - mean1;
            double diff2 = desc2.at<double>(i, j) - mean2;
            sum_sq_diff1 += diff1 * diff1;
            sum_sq_diff2 += diff2 * diff2;
            sum_product += diff1 * diff2;
        }
    }
    double stddev1 = sqrt(sum_sq_diff1 / (desc1.rows * desc1.cols)); // Compute standard deviation of first descriptor
    double stddev2 = sqrt(sum_sq_diff2 / (desc2.rows * desc2.cols)); // Compute standard deviation of second descriptor

    // Compute NCC
    double ncc = sum_product / (stddev1 * stddev2 * desc1.rows * desc1.cols); // Compute normalized cross-correlation
    return ncc; // Return the computed NCC
}

/**
 * Assign orientation to keypoints based on local image gradients.
 *
 * @param pyramid Image pyramid containing different levels of the image.
 * @param keypoints Vector of keypoints to assign orientation.
 */
void assignOrientationToKeypoints(const std::vector<Mat>& pyramid, std::vector<KeyPoint>& keypoints) {
    const int num_bins = 36; // Number of orientation bins
    const double bin_width = 2 * CV_PI / num_bins; // Width of each orientation bin

    for (size_t i = 0; i < keypoints.size(); ++i) {
        KeyPoint& kp = keypoints[i];
        int level = kp.octave; // Level of the keypoint in the image pyramid
        int radius = cvRound(3 * kp.size); // Radius for calculating gradients
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);

        // Ensure the keypoint is within the image boundaries
        if (x < radius || y < radius || x >= pyramid[level].cols - radius || y >= pyramid[level].rows - radius) {
            continue; // Skip keypoints near the image borders
        }

        // Compute gradients in the neighborhood of the keypoint
        Mat grad_x, grad_y;
        try {
            Sobel(pyramid[level], grad_x, CV_32F, 1, 0, 2 * radius + 1); // Compute gradient in x-direction
            Sobel(pyramid[level], grad_y, CV_32F, 0, 1, 2 * radius + 1); // Compute gradient in y-direction
        }
        // Debug lines to avoid any errors occurs
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV exception: " << e.what() << std::endl;
            std::cerr << "KeyPoint index: " << i << std::endl;
            std::cerr << "Level: " << level << ", Radius: " << radius << std::endl;
            std::cerr << "KeyPoint position: (" << x << ", " << y << ")" << std::endl;
            std::cerr << "Image size: " << pyramid[level].cols << "x" << pyramid[level].rows << std::endl;
            continue;
        }

        // Check dimensions of gradient matrices
        std::cout << "grad_x dimensions: " << grad_x.rows << "x" << grad_x.cols << std::endl;
        std::cout << "grad_y dimensions: " << grad_y.rows << "x" << grad_y.cols << std::endl;

        // Calculate gradient magnitude and orientation
        Mat magnitude, angle;
        cartToPolar(grad_x, grad_y, magnitude, angle, true); // Convert gradients to polar coordinates

        // Check dimensions of magnitude and angle matrices
        std::cout << "magnitude dimensions: " << magnitude.rows << "x" << magnitude.cols << std::endl;
        std::cout << "angle dimensions: " << angle.rows << "x" << angle.cols << std::endl;

        // Initialize histogram to accumulate gradient orientations
        std::vector<double> histogram(num_bins, 0.0);

        // Accumulate gradient orientations into histogram bins
        for (int j = -radius; j <= radius; ++j) {
            for (int k = -radius; k <= radius; ++k) {
                // Calculate gradient orientation relative to the keypoint
                double angle_rad = angle.at<float>(y + j, x + k) * CV_PI / 180.0;
                double weight = exp(-(j * j + k * k) / (2 * radius * radius)); // Gaussian weight
                int bin = cvRound(angle_rad / bin_width);
                bin = (bin < 0) ? num_bins + bin : bin; // Handle negative bins
                bin = (bin >= num_bins) ? bin - num_bins : bin; // Handle overflow bins
                histogram[bin] += weight * magnitude.at<float>(y + j, x + k); // Accumulate weighted magnitude
            }
        }

        // Find dominant orientation(s) in the histogram
        double max_value = *std::max_element(histogram.begin(), histogram.end());
        const double threshold_ratio = 0.8; // Threshold ratio for retaining multiple peaks
        for (int b = 0; b < num_bins; ++b) {
            // Check if the bin value exceeds a threshold of the maximum value
            if (histogram[b] >= threshold_ratio * max_value) {
                // Compute orientation angle in degrees
                double angle_deg = (b + 0.5) * bin_width * 180.0 / CV_PI;
                // Create new keypoint with the detected orientation
                KeyPoint new_kp(kp.pt, kp.size, angle_deg, kp.response, kp.octave, kp.class_id);
                keypoints.push_back(new_kp); // Add the new keypoint to the list
            }
        }
    }
}


/**
 * Match descriptors between two sets using Sum of Squared Differences (SSD) or Normalized Cross-Correlation (NCC).
 *
 * @param descriptors1 Descriptors of the first set.
 * @param descriptors2 Descriptors of the second set.
 * @param matches Vector to store the matched descriptors.
 * @param method Method to compute similarity ('SSD' or 'NCC').
 */
void matchDescriptors(const Mat& descriptors1, const Mat& descriptors2, std::vector<DMatch>& matches, std::string method) {
    // Clear existing matches
    matches.clear();

    // Loop through all descriptors in the first set
    for (int i = 0; i < descriptors1.rows; ++i) {
        int best_match_index = -1;
        double best_similarity = -std::numeric_limits<double>::max(); // Initialize with minimum value for SSD, maximum value for NCC

        // Compare the current descriptor with all descriptors in the second set
        for (int j = 0; j < descriptors2.rows; ++j) {
            // Compute similarity score using the specified method
            double similarity;
            if (method == "SSD") {
                similarity = computeSSD(descriptors1.row(i), descriptors2.row(j));
            }
            else if (method == "NCC") {
                similarity = computeNCC(descriptors1.row(i), descriptors2.row(j));
            }
            else {
                std::cerr << "Error: Invalid method specified." << std::endl;
                return;
            }

            // Update best match if a better similarity score is found
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match_index = j;
            }
        }

        // Add the best match to the matches vector
        matches.push_back(DMatch(i, best_match_index, 0)); // Distance is not used for SSD and NCC
    }
}

/**
 * Detect keypoints in each level of the image pyramid using Difference of Gaussians (DoG).
 *
 * @param pyramid The image pyramid containing different levels of the image.
 * @param keypoints Vector to store the detected keypoints.
 * @param response_threshold The minimum response value required for a keypoint to be retained.
 */
void detectKeypointDoG(const std::vector<Mat>& pyramid, std::vector<KeyPoint>& keypoints, double response_threshold) {
    const double threshold = 0.8; // Threshold for detecting keypoints

    // Iterate through each level of the image pyramid
    for (int i = 1; i < pyramid.size() - 1; ++i) {
        Mat DoG;
        absdiff(pyramid[i], pyramid[i + 1], DoG); // Compute Difference of Gaussians

        // Check for extrema in the DoG image
        for (int y = 1; y < DoG.rows - 1; ++y) {
            for (int x = 1; x < DoG.cols - 1; ++x) {
                float center = DoG.at<float>(y, x);
                // Check if the center pixel is an extrema
                if (center > threshold &&
                    center > DoG.at<float>(y - 1, x - 1) && center > DoG.at<float>(y - 1, x) && center > DoG.at<float>(y - 1, x + 1) &&
                    center > DoG.at<float>(y, x - 1) && center >= DoG.at<float>(y, x + 1) &&
                    center >= DoG.at<float>(y + 1, x - 1) && center >= DoG.at<float>(y + 1, x) && center >= DoG.at<float>(y + 1, x + 1)) {
                    // Add keypoint to the vector
                    keypoints.push_back(KeyPoint(x, y, 2 * sqrt(2) * i)); // Scale by 2*sqrt(2) for better scale consistency
                }
                    // Check if the center pixel is a negative extrema
                else if (center < -threshold &&
                         center < DoG.at<float>(y - 1, x - 1) && center < DoG.at<float>(y - 1, x) && center < DoG.at<float>(y - 1, x + 1) &&
                         center < DoG.at<float>(y, x - 1) && center <= DoG.at<float>(y, x + 1) &&
                         center <= DoG.at<float>(y + 1, x - 1) && center <= DoG.at<float>(y + 1, x) && center <= DoG.at<float>(y + 1, x + 1)) {
                    // Add keypoint to the vector
                    keypoints.push_back(KeyPoint(x, y, 2 * sqrt(2) * i)); // Scale by 2*sqrt(2) for better scale consistency
                }
            }
        }
    }

    // Filter keypoints based on response threshold
    filterKeypoints(keypoints, pyramid[0], response_threshold);
}

/**
 * Main SIFT function.
 *
 * @param image1 The first input image.
 * @param image2 The second input image.
 */
cv::Mat sift(const cv::Mat& image1, const cv::Mat& image2, SimilarityMetric metric, int levels, double scale_factor, double sigma, double response_threshold){
    // Parameters
    //int levels = 5; // Number of levels in the image pyramid
    //double scale_factor = 0.5; // Scaling factor for each level
    //double sigma = 1.6; // Standard deviation for Gaussian blur

    // Apply Gaussian blur to input images
    Mat blurred_image1 = applyGaussianBlur(image1, sigma);
    Mat blurred_image2 = applyGaussianBlur(image2, sigma);

    // Build image pyramids
    std::vector<Mat> pyramid1, pyramid2;
    buildImagePyramid(blurred_image1, pyramid1, levels, scale_factor);
    buildImagePyramid(blurred_image2, pyramid2, levels, scale_factor);

    // Detect keypoints
    std::vector<KeyPoint> keypoints1, keypoints2;
    detectKeypointsDOG(pyramid1, keypoints1, 0.0);
    detectKeypointsDOG(pyramid2, keypoints2, 0.0);

    // Assign orientation to keypoints
    //assignOrientationToKeypoints(pyramid1, keypoints1);
    //assignOrientationToKeypoints(pyramid2, keypoints2);

    // Compute descriptors
    Mat descriptors1, descriptors2;
    computeDescriptors(blurred_image1, keypoints1, descriptors1);
    computeDescriptors(blurred_image2, keypoints2, descriptors2);

    metricName = "NCC";

    Mat matched_image;
    std::vector<DMatch> matches_ncc;
    matchDescriptors(descriptors1, descriptors2, matches_ncc, metricName);
    drawMatchesFeatures(image1, keypoints1, image2, keypoints2, matches_ncc, matched_image);

    return matched_image;
}


/**
 * Draw circles around keypoints on the image with random colors.
 *
 * @param image The input image.
 * @param keypoints Keypoints to be drawn.
 * @param output Output image with keypoints drawn.
 */
void drawKeypoints(const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& output) {
    // Draw circles around keypoints on the image with random colors
    output = image.clone();
    RNG rng; // Random number generator
    for (const auto& kp : keypoints) {
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)); // Random color
        circle(output, kp.pt, 5, color, 2); // Circle with radius 5
    }
}

/**
 * Compute keypoints and descriptors from the input image.
 *
 * @param image The input image.
 * @param keypoints Vector to store detected keypoints.
 * @param descriptors Matrix to store computed descriptors.
 * @param response_threshold Threshold for keypoint detection.
 */
Mat computeKeypoints(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors,int levels ,double scale_factor , double sigma ,double response_threshold) {
    // Parameters
    //int levels = 4; // Number of levels in the image pyramid
    //double scale_factor = 0.5; // Scaling factor for each level
    //double sigma = 1.6; // Standard deviation for Gaussian blur
    // Draw keypoints on the image
    Mat image_with_keypoints;
    // Apply Gaussian blur to input image
    Mat blurred_image = applyGaussianBlur(image, sigma);

    // Build image pyramid
    std::vector<Mat> pyramid;
    buildImagePyramid(blurred_image, pyramid, levels, scale_factor);

    // Detect keypoints
    detectKeypointsDOG(pyramid, keypoints, response_threshold);

    // Filter keypoints based on response value
    filterKeypoints(keypoints, blurred_image, response_threshold);

    // Compute descriptors
    computeDescriptors(blurred_image, keypoints, descriptors);
    // Display image with keypoints based on response threshold
    if (response_threshold == 0.001) {
    drawKeypoints(image, keypoints, image_with_keypoints);
    return image_with_keypoints;
    }
    else {
    drawKeypoints(image, keypoints, image_with_keypoints);
    return image_with_keypoints;
    }
}

