#pragma once

#include <iostream>

#include <opencv2/highgui.hpp>

namespace fail_3d {

namespace {
constexpr int kArrowUpKey = 82;
constexpr int kArrowDownKey = 84;
constexpr int kEscapeKey = 27;

}  // anonymous namespace

void VisualizePyramid(const std::vector<cv::Mat>& pyramid) {
  size_t sz = pyramid.size();
  size_t k = 0;
  int x;
  do {
    cv::destroyAllWindows();
    std::string winname = "pyramid " + std::to_string(k);
    cv::imshow(winname, pyramid[k]);
    do {
      x = cv::waitKey(0);
      if (x == kArrowUpKey)  // up arrow
        k = (k + 1) % sz;
      else if (x == kArrowDownKey)  // down arrow
        k = (k > 0) ? (k - 1) : (sz - 1);
    } while (x != kArrowUpKey && x != kArrowDownKey && x != kEscapeKey);
  } while (x != kEscapeKey);
}

}  // namespace fail_3d
