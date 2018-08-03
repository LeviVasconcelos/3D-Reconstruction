#pragma once

#include <eigen3/Eigen/Core>

#include <opencv2/imgproc.hpp>

namespace fail_3d {
struct Resolution {
  int x;
  int y;
  Resolution(int x, int y) : x(x), y(y) {}
};

struct Cell {
  int width;
  int height;
  Cell(int width, int height) : width(width), height(height) {}
};

struct Grid {
  Resolution resolution;
  Cell cell;
  Grid(const Resolution& r, const cv::Size& sz)
      : resolution(r.x, r.y), cell(sz.width / r.x, sz.height / r.y) {}
};

void ComputeCornersOnGrid(const cv::Mat& src, cv::Mat& dst, const Grid& grid);
void GoodFeaturesToTrack(const cv::Mat& src, std::vector<cv::Point2f>& dst);
void WarpImageWithTranslation(const cv::Mat& src, cv::Mat& dst,
                              const cv::Point2f& d);
std::vector<cv::Mat> BuildImagePyramid(const cv::Mat& src, size_t levels);

Eigen::Matrix2f ComputeG(const cv::Mat& x_derivative,
                         const cv::Mat& y_derivative);

Eigen::Vector2f ComputeB(const cv::Mat& x_derivative,
                         const cv::Mat& y_derivative,
                         const cv::Mat& t_derivative);
void ComputeXYImageDerivatives(const cv::Mat& src, cv::Mat& x_derivative,
                               cv::Mat& y_derivative);

void ComputeTimeDerivative(const cv::Mat& t1, const cv::Mat& t2, cv::Mat& out);

void CropRectSubpix(cv::Mat& src, const cv::Size& window,
                    const cv::Point2f& center, cv::Mat& out);
}  // namespace fail_3d
