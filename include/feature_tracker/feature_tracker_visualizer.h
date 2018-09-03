#pragma once

#include <opencv2/core.hpp>

namespace fail_3d {

void VisualizePyramid(const std::vector<cv::Mat>& pyramid);
void VisualizePyramidWithKeypoints(const std::vector<cv::Mat>& pyramid,
                                   const std::vector<cv::Point2f>& points);
void DrawKeypoints(cv::Mat& image, const std::vector<cv::Point2f>& points);
void VisualizeTrackingHistory(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<cv::Point2f>>& points_history);

}  // namespace fail_3d
