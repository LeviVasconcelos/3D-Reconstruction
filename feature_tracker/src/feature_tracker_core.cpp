#include <feature_tracker/feature_tracker_core.h>

#include <feature_tracker/feature_tracker_utils.h>
#include <feature_tracker/feature_tracker_visualizer.h>

#include <eigen3/Eigen/Dense>

namespace fail_3d {

FeatureTracker::FeatureTracker(const size_t track_window_size,
                               const size_t feature_window_size,
                               const size_t pyramid_levels)
    : track_window_size_(track_window_size),
      feature_window_size_(feature_window_size),
      pyramid_levels_(pyramid_levels) {}

void FeatureTracker::Initialize(const cv::Mat &init_frame) {
  Reset();
  GoodFeaturesToTrack(init_frame, positions_to_track_);
  current_pyramid_ = BuildImagePyramid(init_frame, pyramid_levels_);
  ComputePyramidXYDerivatives();
  VisualizePyramid(x_derivatives);
  VisualizePyramid(y_derivatives);
}

void FeatureTracker::Track(const cv::Mat &next_frame) {
  std::vector<cv::Mat> next_pyramid =
      BuildImagePyramid(next_frame, pyramid_levels_);
  std::vector<cv::Point2f> delta(positions_to_track_.size(), cv::Point2f{0, 0});
  // Compute time derivative of images
  VisualizePyramid(next_pyramid);
  VisualizePyramid(current_pyramid_);
  ComputePyramidTimeDerivatives(next_pyramid);
  VisualizePyramid(time_derivatives);
  cv::Size window_size{track_window_size_, track_window_size_};
  for (size_t i = pyramid_levels_; i-- > 0;) {
    for (size_t k = 0; k < positions_to_track_.size(); ++k) {
      if (k == 50) std::cout << "delta before: " << delta[k] << std::endl;
      delta[k] = 2 * delta[k];
      if (k == 50) std::cout << "delta after: " << delta[k] << std::endl;
      cv::Point2f &point_feature = positions_to_track_[k];
      // Get coordinates for pyramid level
      float scaled_x = point_feature.x / pow(2., i);
      float scaled_y = point_feature.y / pow(2., i);

      // Get image window shifted by delta (computed at previous pyramid level)
      cv::Point2f center{scaled_x + delta[k].x, scaled_y + delta[k].y};
      cv::Mat cropped_deriv_x;
      cv::Mat cropped_deriv_y;
      cv::Mat cropped_deriv_t;
      CropRectSubpix(x_derivatives[i], window_size, center, cropped_deriv_x);
      CropRectSubpix(y_derivatives[i], window_size, center, cropped_deriv_y);
      CropRectSubpix(time_derivatives[i], window_size, center, cropped_deriv_t);

      // Compute d = inv(G)*b.
      Eigen::Matrix2f G = ComputeG(cropped_deriv_x, cropped_deriv_y);
      Eigen::Vector2f b =
          ComputeB(cropped_deriv_x, cropped_deriv_y, cropped_deriv_t);
      if (k == 50) std::cout << "G: " << G << std::endl;
      if (k == 50) std::cout << "G_inv: " << G.inverse() << std::endl;
      if (k == 50) std::cout << "b: " << b << std::endl;
      Eigen::Vector2f d = -(G.inverse() * b);
      if (k == 50) std::cout << "computed d: " << d << std::endl;
      delta[k].x += d[0];
      delta[k].y += d[1];
      //      char x;
      //      std::cin >> x;
      //      if (x == 27) break;
      // Check whether d is feasible (out of image boundaries)
    }
    // Save d for refinement
  }

  std::cout << "*** DELTAS ***" << std::endl;
  for (const auto &p : delta) {
    std::cout << "(" << p.x << ", " << p.y << ") ";
  }
  std::cout << std::endl;
}

void FeatureTracker::ComputePyramidXYDerivatives() {
  for (size_t i = 0; i < pyramid_levels_; ++i) {
    ComputeXYImageDerivatives(current_pyramid_[i], x_derivatives[i],
                              y_derivatives[i]);
  }
}

void FeatureTracker::ComputePyramidTimeDerivatives(
    const std::vector<cv::Mat> &target) {
  time_derivatives.clear();
  time_derivatives.reserve(pyramid_levels_);
  for (size_t i = 0; i < current_pyramid_.size(); ++i) {
    cv::Mat diff;
    cv::subtract(target[i], current_pyramid_[i], diff, cv::Mat{}, CV_8UC1);
    time_derivatives.push_back(diff);
  }
}

void FeatureTracker::Reset() {
  positions_to_track_.clear();
  x_derivatives.clear();
  y_derivatives.clear();
  x_derivatives.resize(pyramid_levels_);
  y_derivatives.resize(pyramid_levels_);
}

std::vector<cv::Point2f> FeatureTracker::GetPositions() {
  return positions_to_track_;
}

}  // namespace fail_3d
