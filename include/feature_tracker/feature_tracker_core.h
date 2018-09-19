#pragma once

#include <vector>

#include <eigen3/Eigen/Core>

#include <opencv2/core.hpp>

namespace fail_3d {
class FeatureTracker {
 public:
  FeatureTracker(const size_t track_window_size = 17,
                 const size_t feature_window_size = 17,
                 const size_t pyramid_size = 3);
  void Initialize(const cv::Mat& init_frame);
  void Track(const cv::Mat& next_frame);
  void SetCurrentImage(const cv::Mat& current_image);

  std::vector<cv::Point2f> GetPositions();

 private:
  std::vector<Eigen::Vector2f> TrackImages(
      const cv::Mat& prev, const cv::Mat& current,
      const std::vector<cv::Point2f>& points_to_track);
  void Reset();
  void ComputePyramidXYDerivatives();
  void ComputePyramidTimeDerivatives(const std::vector<cv::Mat>& target);

  std::vector<cv::Point2f> positions_to_track_;
  std::vector<bool> good_features;
  size_t track_window_size_;
  size_t feature_window_size_;
  size_t pyramid_levels_;
  std::vector<cv::Mat> current_pyramid_;
  std::vector<cv::Mat> x_derivatives;
  std::vector<cv::Mat> y_derivatives;
  std::vector<cv::Mat> time_derivatives;
};
}  // namespace fail_3d
