#include <feature_tracker/feature_tracker_utils.h>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

namespace {
constexpr int kBlockSize = 5;
constexpr int kSobelSize = 3;
constexpr double kHarrisThreshold = 0.05;

constexpr int kMaxCorners = 500;
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
                              const cv::Point2f &d) {}

}  // namespace fail_3d
