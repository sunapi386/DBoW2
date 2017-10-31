/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

#define PRINT_COLOR_RED "\033[22;31m"
#define PRINT_COLOR_YELLOW "\033[22;33m"
#define PRINT_COLOR_GRAY "\033[22;90m"
#define PRINT_COLOR_RESET "\033[0m"
#define RR(__O__) std::cerr<<PRINT_COLOR_RED" ERROR: "<<PRINT_COLOR_RESET<<__O__<<std::endl
#define EM(__O__) std::cout<<PRINT_COLOR_YELLOW" INFO: "<<PRINT_COLOR_RESET<<__O__<<std::endl
#define DB(__O__) std::cout<<PRINT_COLOR_GRAY"  DEBUG: "<<PRINT_COLOR_RESET<<__O__<<std::endl


// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <gflags/gflags.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>


using namespace DBoW2;
using namespace DUtils;
using namespace std;
using boost::filesystem::path;
using boost::filesystem::directory_iterator;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeatures(vector<vector<cv::Mat>> features, vector<string> image_paths);

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);

void testVocCreation(vector<vector<cv::Mat>> features, vector<string> image_paths);

void testDatabase(vector<vector<cv::Mat>> features, vector<string> image_paths);

vector<string> listFiles(const string &strpath, const string &ext = "") {
  path p(strpath);
  vector<string> files;
  for (auto i = directory_iterator(p); i != directory_iterator(); i++) {
    if (!is_directory(i->path())) { /*not a directory*/
      string filename = i->path().filename().string();
      const string &file_ext = extension(i->path());
      if(ext.empty() || (!ext.empty() && file_ext == ext)) {
        files.emplace_back(strpath + i->path().preferred_separator + i->path().string());
      }
    }
  }
  return files;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait() {
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// gflags
/////////////////////////////////////////////////////////////////////////////////////////////////////
DEFINE_string(dir, "", "directory to jpg images");
DEFINE_int32(branching_level, 9, "internal: k-d tree branch level");
DEFINE_int32(depth_factors, 3, "internal: k-d tree depth factor");

static bool ValidatePathIsDirectory(const char *flag, const std::string &path) {
  if (path.empty()) {
    return false; // assume current working directory
  }

  bool success = false;
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0) {
    success = (buffer.st_mode & S_IFDIR) == S_IFDIR; // check for directory mask in st_mode
  }
  if (!success) {
    RR("path is not a directory: " << path);
  }

  return success;
}

static bool ValidateFileExists(const char *flag, const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

DEFINE_validator(dir, &ValidatePathIsDirectory);

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// main
/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  const char *ext = ".jpg";
  auto image_paths = listFiles(FLAGS_dir, ext);
  EM("Found " << image_paths.size() << " " << ext << " files");
  for(auto &file : image_paths) {
    EM(file);
  }

  vector<vector<cv::Mat> > features;
  loadFeatures(features, image_paths);

  testVocCreation(features, image_paths);

  wait();

  testDatabase(features, image_paths);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat>> features, vector<string> image_paths) {
  features.clear();
  features.reserve(image_paths.size());

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (auto &file : image_paths) {
    cv::Mat image = cv::imread(file, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) {
  out.resize(plain.rows);

  for (int i = 0; i < plain.rows; ++i) {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(vector<vector<cv::Mat>> features, vector<string> image_paths) {
  // branching factor and depth levels 
  const int k = FLAGS_branching_level;
  const int L = FLAGS_depth_factors;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  OrbVocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for (int i = 0; i < image_paths.size(); i++) {
    voc.transform(features[i], v1);
    for (int j = 0; j < image_paths.size(); j++) {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(vector<vector<cv::Mat>> features, vector<string> image_paths) {
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");

  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (int i = 0; i < image_paths.size(); i++) {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for (int i = 0; i < image_paths.size(); i++) {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


