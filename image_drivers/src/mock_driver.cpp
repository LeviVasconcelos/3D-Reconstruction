#include <image_drivers/mock_driver.h>

// Opencv includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// 3d_fail includes
#include <io/io.h>

namespace fail_3d {

MockImageDriver::MockImageDriver(const std::string &path_dir) {
  LoadImageFilePaths(path_dir);
  LoadNextImage();
}

void MockImageDriver::LoadImageFilePaths(const std::string &path_dir) {
  images_file_paths = io::LoadFilesWithExtension(path_dir, io::image_types);
  index = 0;
}

bool MockImageDriver::LoadNextImage() {
  if (index < images_file_paths.size()) {
    current_image =
        cv::imread(images_file_paths[index++], CV_LOAD_IMAGE_GRAYSCALE);
    return true;
  } else {
    LOG("LOAD_NEXT_IMAGE") << "index out of range." << std::endl;
    return false;
  }
}

cv::Mat &MockImageDriver::GetImage() { return current_image; }

}  // namespace fail_3d
