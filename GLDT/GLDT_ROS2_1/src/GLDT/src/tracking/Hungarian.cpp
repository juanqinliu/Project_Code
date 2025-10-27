#include "tracking/Hungarian.h"
#include <limits>
#include <algorithm>
#include <iostream>

namespace tracking {

cv::Mat padToSquare(const cv::Mat& matrix, float pad_value) {
    int rows = matrix.rows;
    int cols = matrix.cols;
    int size = std::max(rows, cols);
    
    cv::Mat padded_matrix(size, size, CV_32F, cv::Scalar(pad_value));
    
    // Copy original matrix to top-left corner
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix.type() == CV_8U) {
                padded_matrix.at<float>(i, j) = static_cast<float>(matrix.at<unsigned char>(i, j)) / 255.0f;
            } else if (matrix.type() == CV_32F) {
                padded_matrix.at<float>(i, j) = matrix.at<float>(i, j);
            }
        }
    }
    
    return padded_matrix;
}

std::vector<std::pair<int, int>> assignmentToMatches(const cv::Mat& assignment) {
    std::vector<std::pair<int, int>> matches;
    
    for (int i = 0; i < assignment.rows; i++) {
        for (int j = 0; j < assignment.cols; j++) {
            if (assignment.at<unsigned char>(i, j) == 1) {
                matches.emplace_back(i, j);
            }
        }
    }
    
    return matches;
}

float Hungarian(const cv::Mat& cost_matrix, cv::Mat& assignment) {
    int rows = cost_matrix.rows;
    int cols = cost_matrix.cols;
    
    // Ensure matrix is square
    int size = std::max(rows, cols);
    cv::Mat cost(size, size, CV_32F, cv::Scalar(1000.0f)); // High value for non-matching
    
    // Copy cost matrix and convert to float
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (cost_matrix.type() == CV_8U) {
                cost.at<float>(i, j) = static_cast<float>(cost_matrix.at<unsigned char>(i, j)) / 255.0f;
            } else if (cost_matrix.type() == CV_32F) {
                cost.at<float>(i, j) = cost_matrix.at<float>(i, j);
            }
        }
    }
    
    // Initialize assignment matrix
    assignment = cv::Mat::zeros(rows, cols, CV_8U);
    
    // Hungarian algorithm variables
    std::vector<int> row_cover(size, 0);
    std::vector<int> col_cover(size, 0);
    std::vector<int> star_matrix(size * size, 0);
    std::vector<int> prime_matrix(size * size, 0);
    std::vector<int> new_star_matrix(size * size, 0);
    
    // Step 1: Subtract row minimum from each row
    for (int i = 0; i < size; i++) {
        float min_val = cost.at<float>(i, 0);
        for (int j = 1; j < size; j++) {
            if (cost.at<float>(i, j) < min_val) {
                min_val = cost.at<float>(i, j);
            }
        }
        for (int j = 0; j < size; j++) {
            cost.at<float>(i, j) -= min_val;
        }
    }
    
    // Step 2: Find zeros and star them
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (cost.at<float>(i, j) < 1e-6f && row_cover[i] == 0 && col_cover[j] == 0) {
                star_matrix[i * size + j] = 1;
                row_cover[i] = 1;
                col_cover[j] = 1;
            }
        }
    }
    
    // Reset covers
    std::fill(row_cover.begin(), row_cover.end(), 0);
    std::fill(col_cover.begin(), col_cover.end(), 0);
    
    // Main Hungarian loop
    int step = 3;
    bool done = false;
    int row = -1, col = -1;
    
    while (!done) {
        switch (step) {
            case 3: {
                // Cover columns with starred zeros
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < size; i++) {
                        if (star_matrix[i * size + j] == 1) {
                            col_cover[j] = 1;
                            break;
                        }
                    }
                }
                
                // Check if all columns are covered
                int col_count = 0;
                for (int j = 0; j < size; j++) {
                    if (col_cover[j] == 1) {
                        col_count++;
                    }
                }
                
                step = (col_count >= size) ? 7 : 4;
                break;
            }
            
            case 4: {
                // Find uncovered zero
                bool found = false;
                for (int i = 0; i < size && !found; i++) {
                    for (int j = 0; j < size && !found; j++) {
                        if (cost.at<float>(i, j) < 1e-6f && row_cover[i] == 0 && col_cover[j] == 0) {
                            prime_matrix[i * size + j] = 1;
                            
                            // Check for star in row
                            bool star_in_row = false;
                            for (int j2 = 0; j2 < size; j2++) {
                                if (star_matrix[i * size + j2] == 1) {
                                    star_in_row = true;
                                    col = j2;
                                    break;
                                }
                            }
                            
                            if (!star_in_row) {
                                row = i;
                                col = j;
                                step = 5;
                                found = true;
                            } else {
                                row_cover[i] = 1;
                                col_cover[col] = 0;
                            }
                        }
                    }
                }
                
                if (!found) {
                    step = 6;
                }
                break;
            }
            
            case 5: {
                // Construct augmenting path
                int path_count = 0;
                new_star_matrix = star_matrix;
                
                // Add prime zero
                new_star_matrix[row * size + col] = 1;
                
                // Find alternating path
                bool done5 = false;
                while (!done5) {
                    // Find star in column
                    bool star_in_col = false;
                    for (int i = 0; i < size; i++) {
                        if (star_matrix[i * size + col] == 1) {
                            row = i;
                            star_in_col = true;
                            break;
                        }
                    }
                    
                    if (star_in_col) {
                        // Remove star
                        new_star_matrix[row * size + col] = 0;
                        
                        // Find prime in row
                        for (int j = 0; j < size; j++) {
                            if (prime_matrix[row * size + j] == 1) {
                                col = j;
                                new_star_matrix[row * size + col] = 1;
                                break;
                            }
                        }
                    } else {
                        done5 = true;
                    }
                    
                    path_count++;
                    if (path_count > size * size) {
                        done5 = true; // Prevent infinite loop
                    }
                }
                
                // Update star matrix
                star_matrix = new_star_matrix;
                
                // Clear primes and covers
                std::fill(prime_matrix.begin(), prime_matrix.end(), 0);
                std::fill(row_cover.begin(), row_cover.end(), 0);
                std::fill(col_cover.begin(), col_cover.end(), 0);
                
                step = 3;
                break;
            }
            
            case 6: {
                // Find minimum uncovered value
                float min_val = std::numeric_limits<float>::max();
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (row_cover[i] == 0 && col_cover[j] == 0 && cost.at<float>(i, j) < min_val) {
                            min_val = cost.at<float>(i, j);
                        }
                    }
                }
                
                // Update cost matrix
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (row_cover[i] == 1) {
                            cost.at<float>(i, j) += min_val;
                        }
                        if (col_cover[j] == 0) {
                            cost.at<float>(i, j) -= min_val;
                        }
                    }
                }
                
                step = 4;
                break;
            }
            
            case 7:
                done = true;
                break;
        }
    }
    
    // Build final assignment
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i < size && j < size && star_matrix[i * size + j] == 1) {
                assignment.at<unsigned char>(i, j) = 1;
            }
        }
    }
    
    // Calculate minimum cost
    float min_cost = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (assignment.at<unsigned char>(i, j) == 1) {
                if (cost_matrix.type() == CV_8U) {
                    min_cost += static_cast<float>(cost_matrix.at<unsigned char>(i, j)) / 255.0f;
                } else if (cost_matrix.type() == CV_32F) {
                    min_cost += cost_matrix.at<float>(i, j);
                }
            }
        }
    }
    
    return min_cost;
}

} // namespace tracking 