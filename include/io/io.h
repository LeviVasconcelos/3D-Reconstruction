#pragma once
// System includes
#include <iostream>
#include <string>
#include <vector>

// Boost includes
#include <boost/filesystem.hpp>

// TODO: input line and file name
#define LOG(x) std::cout << "(LOG)[" << x << "]"
namespace fail_3d {
namespace io {

namespace fs = boost::filesystem;
const std::vector<std::string> image_types = {"jpg", "png", "jpeg"};

std::vector<std::string> LoadFilesWithExtension(
    const std::string& path, const std::vector<std::string>& extensions);

}  // namespace io
}  // namespace fail_3d
