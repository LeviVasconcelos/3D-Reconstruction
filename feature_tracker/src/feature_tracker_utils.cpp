#include <feature_tracker/feature_tracker_utils.h>

#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

namespace {
constexpr int kBlockSize = 5;
constexpr int kSobelSize = 3;
constexpr double kHarrisThreshold = 0.05;

constexpr int kMaxCorners = 300;
constexpr double kQualityLevel = 0.01;
constexpr double kMinDist = 15;
constexpr bool kUseHarris = true;

}  // namespace anonymous

namespace fail_3d {

void ComputeCornersOnGrid(const cv::Mat &src, cv::Mat &dst, const Grid &grid) {
  dst.create(src.size(), CV_32FC1);
  for (int i = 0; i < grid.resolution.x; ++i) {
    for (int j = 0; j < grid.resolution.y; ++j) {
      cv::Point2i cell_center(i * (grid.cell.width) + (grid.cell.width / 2.),
                              j * (grid.cell.height) + (grid.cell.height / 2.));
      cv::Point2i top_left(cell_center.x - (grid.cell.width / 2.),
                           cell_center.y - (grid.cell.height / 2.));
      cv::Rect cell_rect{top_left, cv::Size{grid.cell.width, grid.cell.height}};
      cv::Mat cell = src(cell_rect);
      cv::Mat dst_cell = dst(cell_rect);
      cv::cornerHarris(cell, dst_cell, kBlockSize, kSobelSize,
                       kHarrisThreshold);
    }
  }
  /// Normalizing
  cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
}

void GoodFeaturesToTrack(const cv::Mat &src, std::vector<cv::Point2f> &dst) {
  cv::goodFeaturesToTrack(src, dst, kMaxCorners, kQualityLevel, kMinDist,
                          cv::Mat(), kBlockSize, kSobelSize, kUseHarris);
  cv::cornerSubPix(
      src, dst, cv::Size(kBlockSize, kBlockSize), cv::Size(-1, -1),
      cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
}

void WarpImageWithTranslation(const cv::Mat &src, cv::Mat &dst,
                              const cv::Point2f &d) {
  cv::Mat map_x(src.rows, src.cols, CV_32FC1);
  cv::Mat map_y(src.rows, src.cols, CV_32FC1);
  dst.create(src.rows, src.cols, src.type());
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      map_x.at<float>(i, j) = j + d.x;
      map_y.at<float>(i, j) = i + d.y;
    }
  }
  cv::remap(src, dst, map_x, map_y, CV_INTER_LINEAR);
}

std::vector<cv::Mat> BuildImagePyramid(const cv::Mat &src, size_t levels) {
  std::vector<cv::Mat> pyramid;
  pyramid.reserve(levels);
  pyramid.push_back(src);
  cv::Mat tmp = src;
  for (size_t i = 1; i < levels; ++i) {
    cv::Mat dst;
    cv::pyrDown(tmp, dst, cv::Size(tmp.cols / 2, tmp.rows / 2));
    tmp = dst;
    pyramid.push_back(tmp);
  }
  return pyramid;
}

void ComputeXYImageDerivatives(const cv::Mat &src, cv::Mat &x_derivative,
                               cv::Mat &y_derivative) {
  cv::Mat x_deriv;
  cv::Mat y_deriv;
  cv::Mat src_;
  src.convertTo(src_, CV_32FC1);
  cv::Sobel(src_, x_derivative, CV_32FC1, 1, 0, 3);
  cv::Sobel(src_, y_derivative, CV_32FC1, 0, 1, 3);
  //  std::cout << (int)src.at<uchar>(0, 0) << " " << src_.at<float>(0, 0) << "
  //  "
  //            << x_derivative.at<float>(0, 0) << " "
  //            << y_derivative.at<float>(0, 0) << std::endl;

  // convertScaleAbs(x_deriv, x_derivative);
  // convertScaleAbs(y_deriv, y_derivative);
}

void ComputeTimeDerivative(const cv::Mat &t1, const cv::Mat &t2, cv::Mat &out) {
  cv::subtract(t2, t1, out);
}

Eigen::Matrix2f ComputeG(const cv::Mat &x_derivative,
                         const cv::Mat &y_derivative) {
  assert(x_derivative.cols == y_derivative.cols &&
         x_derivative.rows == y_derivative.rows);
  Eigen::Matrix2f G;
  float x_term = 0;
  float y_term = 0;
  float xy_term = 0;
  for (size_t i = 0; i < x_derivative.rows; ++i) {
    for (size_t j = 0; j < x_derivative.cols; ++j) {
      const float &x_el = x_derivative.at<float>(i, j);
      const float &y_el = y_derivative.at<float>(i, j);
      const float x_el_f = static_cast<float>(x_el);
      const float y_el_f = static_cast<float>(y_el);
      //      std::cout << "derivs_x_el_f*: " << x_el_f << std::endl;
      //      std::cout << "derivs_y_el_f*: " << y_el_f << std::endl;
      x_term += x_el_f * x_el_f;
      y_term += y_el_f * y_el_f;
      xy_term += x_el_f * y_el_f;
    }
  }
  //  std::cout << x_term << " " << xy_term << std::endl
  //            << xy_term << " " << y_term << std::endl;
  G << x_term, xy_term, xy_term, y_term;
  return G;
}

Eigen::Vector2f ComputeB(const cv::Mat &x_derivative,
                         const cv::Mat &y_derivative,
                         const cv::Mat &t_derivative) {
  float x_term = 0;
  float y_term = 0;
  for (size_t i = 0; i < x_derivative.rows; ++i) {
    for (size_t j = 0; j < x_derivative.cols; ++j) {
      const float &x_el = x_derivative.at<float>(i, j);
      const float &y_el = y_derivative.at<float>(i, j);
      const uchar &t_el = t_derivative.at<uchar>(i, j);
      //      std::cout << "x_el: " << x_el << std::endl;
      //      std::cout << "y_el: " << y_el << std::endl;
      const float x_el_f = static_cast<float>(x_el);
      const float y_el_f = static_cast<float>(y_el);
      const float t_el_f = static_cast<float>(t_el);
      //      std::cout << "derivs_x_el_f: " << x_el_f << std::endl;
      //      std::cout << "derivs_y_el_f: " << y_el_f << std::endl;
      //      std::cout << "t_el: " << t_el_f << std::endl;
      x_term += x_el_f * t_el_f;
      y_term += y_el_f * t_el_f;
    }
  }
  // std::cout << "Computed b-terms: " << x_term << " " << y_term << std::endl;
  return {x_term, y_term};
}

void CropRectSubpix(const cv::Mat &src, const cv::Size &window_size,
                    const cv::Point2f &center, cv::Mat &out) {
  //  std::cout << "coords: (" << center.x << " , " << center.y << ") "
  //            << "{" << src.cols << ", " << src.rows << "}" << std::endl;
  assert(src.cols > center.x);
  assert(src.rows > center.y);
  assert(center.x >= 0);
  assert(center.y >= 0);
  out.create(window_size, src.type());
  cv::getRectSubPix(src, window_size, center, out);
  //  cv::imshow("cropped", out);
  //  cv::waitKey(0);
}

cv::Mat TranslateImage(const cv::Mat &input_img, const float x, const float y) {
  /* Build affine matrix of type:
   * [ 1  0  x ]
   * [ 0  1  y ]
   */
  cv::Mat out(input_img.size(), input_img.type());
  cv::Mat affine(2, 3, CV_32FC1);
  affine.at<float>(0, 0) = 1;
  affine.at<float>(0, 1) = 0;
  affine.at<float>(1, 0) = 0;
  affine.at<float>(1, 1) = 1;
  affine.at<float>(0, 2) = x;
  affine.at<float>(1, 2) = y;
  cv::warpAffine(input_img, out, affine, out.size());
  return out;
}

std::vector<Eigen::Vector2f> TrackImages(
    const cv::Mat &prev, const cv::Mat &current,
    const std::vector<cv::Point2f> &points_to_track) {
  const cv::Size kWindowSize{11, 11};
  const float kMinimumShiftLength = 0.001;
  const int kMaxCount = 155;

  // Compute XY derivatives
  cv::Mat prev_x_deriv;
  cv::Mat prev_y_deriv;
  ComputeXYImageDerivatives(prev, prev_x_deriv, prev_y_deriv);

  // Compute deslocation in each tracked feature:
  std::vector<Eigen::Vector2f> shifts;
  shifts.reserve(points_to_track.size());
  for (const auto &run_feature : points_to_track) {
    // Crop windows
    cv::Mat cropped_x_deriv;
    cv::Mat cropped_y_deriv;
    CropRectSubpix(prev_x_deriv, kWindowSize, run_feature, cropped_x_deriv);
    CropRectSubpix(prev_y_deriv, kWindowSize, run_feature, cropped_y_deriv);
    cv::Mat cropped_prev;
    CropRectSubpix(prev, kWindowSize, run_feature, cropped_prev);
    float shift_length = kMinimumShiftLength + 1;
    int count = 0;
    Eigen::Vector2f shift{0., 0.};
    while (shift_length > kMinimumShiftLength && count < kMaxCount) {
      cv::Mat cropped_curr;
      cv::Point2f shift_cv{shift[0], shift[1]};
      CropRectSubpix(current, kWindowSize, run_feature + shift_cv,
                     cropped_curr);
      cv::Mat cropped_time_deriv;
      ComputeTimeDerivative(cropped_prev, cropped_curr, cropped_time_deriv);
      // Compute G and b
      Eigen::Matrix2f G = ComputeG(cropped_x_deriv, cropped_y_deriv);
      Eigen::Matrix2f G_inv = G.inverse();
      Eigen::Vector2f b =
          ComputeB(cropped_x_deriv, cropped_y_deriv, cropped_time_deriv);
      // Compute shift d
      Eigen::Vector2f d = (-G_inv * b);
      shift_length = d.norm();
      shift += d;
      count++;
    }
    shifts.push_back(shift);
  }
  return shifts;
}

std::vector<Eigen::Vector2f> TrackTest(const cv::Mat &prev,
                                       const cv::Mat &current) {
  std::cout << "Hi five, i am being called! " << std::endl;
  const cv::Size kWindowSize{13, 13};
  const float kMinimumShiftLength = 0.01;
  const int kMaxCount = 1000;
  // Find good features to track
  std::vector<cv::Point2f> points_to_track;
  GoodFeaturesToTrack(prev, points_to_track);
  std::cout << "Features found: " << points_to_track.size() << std::endl;

  // Compute XY derivatives
  cv::Mat prev_x_deriv;
  cv::Mat prev_y_deriv;
  ComputeXYImageDerivatives(prev, prev_x_deriv, prev_y_deriv);
  std::cout << "XY derivatives computed." << std::endl;

  // Compute deslocation in each tracked feature:
  std::vector<Eigen::Vector2f> shifts;
  shifts.reserve(points_to_track.size());
  std::cout << "computing shifts!" << std::endl;
  for (const auto &run_feature : points_to_track) {
    // Crop windows
    cv::Mat cropped_x_deriv;
    cv::Mat cropped_y_deriv;
    CropRectSubpix(prev_x_deriv, kWindowSize, run_feature, cropped_x_deriv);
    CropRectSubpix(prev_y_deriv, kWindowSize, run_feature, cropped_y_deriv);
    cv::Mat cropped_prev;
    CropRectSubpix(prev, kWindowSize, run_feature, cropped_prev);
    float shift_length = kMinimumShiftLength + 1;
    int count = 0;
    Eigen::Vector2f shift{0., 0.};
    while (shift_length > kMinimumShiftLength && count < kMaxCount) {
      cv::Mat cropped_curr;
      cv::Point2f shift_cv{shift[0], shift[1]};
      CropRectSubpix(current, kWindowSize, run_feature + shift_cv,
                     cropped_curr);
      cv::Mat cropped_time_deriv;
      ComputeTimeDerivative(cropped_prev, cropped_curr, cropped_time_deriv);
      // cv::subtract(cropped_curr, cropped_prev, cropped_time_deriv);
      // Compute G and b
      Eigen::Matrix2f G = ComputeG(cropped_x_deriv, cropped_y_deriv);
      Eigen::Matrix2f G_inv = G.inverse();
      Eigen::Vector2f b =
          ComputeB(cropped_x_deriv, cropped_y_deriv, cropped_time_deriv);
      // Compute shift d
      Eigen::Vector2f d = (-G_inv * b);
      shift_length = d.norm();
      shift += d;
      count++;
    }
    shifts.push_back(shift);
  }
  // Print shifts
  std::cout << "******** Shifts ********" << std::endl;
  for (const auto &run_shift : shifts) {
    std::cout << "[" << run_shift[0] << ", " << run_shift[1] << "]"
              << std::endl;
  }
  return shifts;
}

}  // namespace fail_3d
