#include <io/io.h>
// System includes
#include <algorithm>

namespace fail_3d {
namespace io {

std::vector<std::string> LoadFilesWithExtension(
    const std::string &path, const std::vector<std::string> &extensions) {
  fs::directory_iterator end_itr;
  std::vector<std::string> paths;
  for (fs::directory_iterator itr(fs::path{path}); itr != end_itr; ++itr) {
    if (fs::is_regular_file(itr->path())) {
      const std::string ext = itr->path().extension().string();
      auto ext_itr = std::find(extensions.begin(), extensions.end(), ext);
      if (ext_itr != image_types.end()) {
        paths.push_back(itr->path().string());
      }
    }
  }
  return paths;
}

}  // namespace io
}  // namespace fail_3d
