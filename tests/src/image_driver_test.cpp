#include <image_drivers/mock_driver.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

int main() {
  fail_3d::MockImageDriver img_driver(
      "/home/levi/workspace/3d_fail/data/images");
  int x;
  do {
    cv::Mat mat = img_driver.GetImage();
    cv::imshow("test", mat);
    x = cv::waitKey(0);
  } while (img_driver.LoadNextImage() && x != 27);
  return 0;
}
