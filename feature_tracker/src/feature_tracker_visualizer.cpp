#include <feature_tracker/feature_tracker_visualizer.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace fail_3d {
namespace {

constexpr int kCircleRadius = 7;
constexpr int kArrowUpKey = 82;
constexpr int kArrowDownKey = 84;
constexpr int kEscapeKey = 27;

}  // annonymous namespace

void VisualizePyramid(const std::vector<cv::Mat>& pyramid) {
  const size_t sz = pyramid.size();
  size_t k = 0;
  int x;
  do {
    std::string winname = "pyramid ";
    std::string title = winname + std::to_string(k);
    cv::imshow(winname, pyramid[k]);
    cv::setWindowTitle(winname, title);
    do {
      x = cv::waitKey(0);
      if (x == kArrowUpKey)  // up arrow
        k = (k + 1) % sz;
      else if (x == kArrowDownKey)  // down arrow
        k = (k > 0) ? (k - 1) : (sz - 1);
    } while (x != kArrowUpKey && x != kArrowDownKey && x != kEscapeKey);
  } while (x != kEscapeKey);
}

void DrawKeypoints(cv::Mat& image, const std::vector<cv::Point2f>& points) {
  for (const auto& p : points) {
    cv::circle(image, p, kCircleRadius, cv::Scalar(0, 0, 0));
  }
}

void VisualizePyramidWithKeypoints(const std::vector<cv::Mat>& pyramid,
                                   const std::vector<cv::Point2f>& points) {
  const size_t sz = pyramid.size();
  size_t k = 0;
  int x;
  do {
    cv::destroyAllWindows();
    std::string winname = "pyramid " + std::to_string(k);
    cv::Mat visualization;
    pyramid[k].copyTo(visualization);
    std::vector<cv::Point2f> new_points;
    new_points.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      const cv::Point2f& ref_point = points[i];
      new_points.emplace_back(ref_point.x / pow(2, k), ref_point.y / pow(2, k));
    }
    DrawKeypoints(visualization, new_points);
    cv::imshow(winname, visualization);
    do {
      x = cv::waitKey(0);
      if (x == kArrowUpKey)  // up arrow
        k = (k + 1) % sz;
      else if (x == kArrowDownKey)  // down arrow
        k = (k > 0) ? (k - 1) : (sz - 1);
    } while (x != kArrowUpKey && x != kArrowDownKey && x != kEscapeKey);
  } while (x != kEscapeKey);
}

void VisualizeTrackingHistory(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<cv::Point2f>>& points_history) {
  std::cout << images.size() << " | " << points_history.size() << std::endl;
  std::vector<cv::Mat> visualization_vector;
  visualization_vector.reserve(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat tmp = images[i];
    visualization_vector.push_back(tmp);
    for (const auto& point : points_history[i]) {
      cv::circle(visualization_vector[i], point, 2, cv::Scalar(0, 0, 0), -1);
    }
  }
  VisualizePyramid(visualization_vector);

  cv::Mat visualization;
  images.back().copyTo(visualization);
  for (const auto& vector_points : points_history) {
    for (const auto& point : vector_points) {
      cv::circle(visualization, point, 2, cv::Scalar(0, 0, 0), -1);
    }
  }
  cv::imshow("Tracking history", visualization);
  cv::waitKey(0);
}

}  // namespace fail_3d
