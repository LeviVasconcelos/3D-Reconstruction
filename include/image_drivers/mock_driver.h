#pragma once

// System includes
#include <string.h>
#include <vector>

// Opencv includes
#include <opencv2/core.hpp>

namespace fail_3d {

class MockImageDriver {
 public:
  MockImageDriver() = delete;
  explicit MockImageDriver(const std::string& path_dir);

  bool LoadNextImage();
  cv::Mat& GetImage();

 private:
  void LoadImageFilePaths(const std::string& path_dir);

  size_t index;
  std::vector<std::string> images_file_paths;
  cv::Mat current_image;
};

}  // fail_3d namespace
