#include <feature_tracker/feature_tracker_core.h>
#include <feature_tracker/feature_tracker_utils.h>
#include <feature_tracker/feature_tracker_visualizer.h>
#include <image_drivers/mock_driver.h>

int main() {
  fail_3d::MockImageDriver img_driver(
      "/home/levi/workspace/3d_fail/data/images");
  cv::Mat init_image;
  img_driver.GetImage().copyTo(init_image);
  cv::Mat new_image = fail_3d::TranslateImage(init_image, -1.5, 2.2);
  fail_3d::TrackTest(init_image, new_image);

  //  fail_3d::FeatureTracker tracker;
  //  tracker.Initialize(init_image);
  //  constexpr int n_images = 2;
  //  std::vector<std::vector<cv::Point2f>> points;
  //  std::vector<cv::Mat> images;
  //  points.reserve(n_images);
  //  images.reserve(n_images);
  //  for (int i = 0; i < n_images; ++i) {
  //    points.push_back(tracker.GetPositions());
  //    cv::Mat tmp;
  //    img_driver.GetImage().copyTo(tmp);
  //    images.push_back(tmp);
  //    img_driver.LoadNextImage();
  //    cv::Mat next_image = img_driver.GetImage();
  //    tracker.Track(next_image);
  //    tracker.SetCurrentImage(next_image);
  //  }
  //  fail_3d::VisualizeTrackingHistory(images, points);
}
