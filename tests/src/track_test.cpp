#include <feature_tracker/feature_tracker_core.h>
#include <image_drivers/mock_driver.h>

int main() {
  fail_3d::MockImageDriver img_driver(
      "/home/levi/workspace/3d_fail/data/images");
  cv::Mat init_image = img_driver.GetImage();
  for (int i = 0; i < 20; ++i) img_driver.LoadNextImage();
  cv::Mat next_image = img_driver.GetImage();
  fail_3d::FeatureTracker tracker;
  tracker.Initialize(init_image);
  std::vector<cv::Point2f> init_points = tracker.GetPositions();
  tracker.Track(next_image);
  std::vector<cv::Point2f> new_points = tracker.GetPositions();
}
