#include <feature_tracker/feature_tracker_utils.h>
#include <image_drivers/mock_driver.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

int main() {
  fail_3d::MockImageDriver img_driver(
      "/home/levi/workspace/3d_fail/tests/data/harris");
  cv::Mat img = img_driver.GetImage();
  std::vector<cv::Point2f> corners;

  fail_3d::GoodFeaturesToTrack(img, corners);
  for (const auto& p : corners) {
    cv::circle(img, p, 5, cv::Scalar(0), 2, 8);
  }
  // fail_3d::Grid grid({5, 5}, img.size());
  // fail_3d::ComputeCornersOnGrid(img, corners, grid);
  //  for (int i = 0; i < img.rows; ++i)
  //    for (int j = 0; j < img.cols; ++j) {
  //      if ((int)corners.at<float>(i, j) > 150)
  //        cv::circle(img, cv::Point(j, i), 5, cv::Scalar(0), 2, 8);
  //    }
  cv::imshow("corners", img);
  cv::waitKey(0);
  return 0;
}
