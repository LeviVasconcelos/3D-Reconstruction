#pragma once
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
  Grid(const Resolution &r, const cv::Size &sz)
      : resolution(r.x, r.y), cell(sz.width / r.x, sz.height / r.y) {}
};

void ComputeCornersOnGrid(const cv::Mat &src, cv::Mat &dst, const Grid &grid);
void GoodFeaturesToTrack(const cv::Mat &src, std::vector<cv::Point2f> &dst);
void WarpImageWithTranslation(const cv::Mat &src, cv::Mat &dst,
                              const cv::Point2f &d);
}
