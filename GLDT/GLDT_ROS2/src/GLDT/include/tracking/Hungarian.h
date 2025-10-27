#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace tracking {

/**
 * @brief Hungarian algorithm implementation for solving assignment problems
 * 
 * The Hungarian algorithm finds the optimal assignment that minimizes the total cost
 * in a bipartite graph. This implementation works with OpenCV Mat objects.
 * 
 * @param cost_matrix Input cost matrix (CV_32F or CV_8U)
 * @param assignment Output assignment matrix (CV_8U, 1 indicates assignment)
 * @return float The minimum total cost of the optimal assignment
 */
float Hungarian(const cv::Mat& cost_matrix, cv::Mat& assignment);

/**
 * @brief Helper function to ensure matrix is square by padding with high cost values
 * 
 * @param matrix Input matrix to be padded
 * @param pad_value Value to use for padding
 * @return cv::Mat Square matrix with padding
 */
cv::Mat padToSquare(const cv::Mat& matrix, float pad_value = 1000.0f);

/**
 * @brief Convert assignment matrix to vector of matched pairs
 * 
 * @param assignment Assignment matrix from Hungarian algorithm
 * @return std::vector<std::pair<int, int>> Vector of (row, col) pairs representing assignments
 */
std::vector<std::pair<int, int>> assignmentToMatches(const cv::Mat& assignment);

} // namespace tracking