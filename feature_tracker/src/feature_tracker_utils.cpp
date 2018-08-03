#include <feature_tracker/feature_tracker_utils.h>

#include <iostream>

#include <eigen3/Eigen/Core>

#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

namespace {
constexpr int kBlockSize = 5;
constexpr int kSobelSize = 3;
constexpr double kHarrisThreshold = 0.05;

constexpr int kMaxCorners = 100;
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
  cv::Sobel(src, x_deriv, CV_16S, 1, 0, 3);
  cv::Sobel(src, y_deriv, CV_16S, 0, 1, 3);
  convertScaleAbs(x_deriv, x_derivative);
  convertScaleAbs(y_deriv, y_derivative);
}

void ComputeTimeDerivative(const cv::Mat &t1, const cv::Mat &t2, cv::Mat &out) {
  out = t2 - t1;
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
      const uchar x_el = x_derivative.at<uchar>(i, j);
      const uchar &y_el = y_derivative.at<uchar>(i, j);
      const float x_el_f = static_cast<float>(x_el);
      const float y_el_f = static_cast<float>(y_el);
      //      std::cout << "derivs_x_el_f*: " << x_el_f << std::endl;
      //      std::cout << "derivs_y_el_f*: " << y_el_f << std::endl;
      x_term += x_el_f * x_el_f;
      y_term += y_el_f * y_el_f;
      xy_term += x_el_f * y_el_f;
    }
  }
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
      const uchar &x_el = x_derivative.at<uchar>(i, j);
      const uchar &y_el = y_derivative.at<uchar>(i, j);
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
  return {x_term, y_term};
}

void CropRectSubpix(cv::Mat &src, const cv::Size &window_size,
                    const cv::Point2f &center, cv::Mat &out) {
  //  std::cout << "coords: (" << center.x << " , " << center.y << ") "
  //            << "{" << src.cols << ", " << src.rows << "}" << std::endl;
  assert(src.cols > center.x);
  assert(src.rows > center.y);
  assert(center.x >= 0);
  assert(center.y >= 0);
  out.create(window_size, src.type());
  cv::getRectSubPix(src, window_size, center, out);
}

}  // namespace fail_3d
