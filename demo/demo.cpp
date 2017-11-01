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
#define WN(__O__) std::cerr<<PRINT_COLOR_RED" WARN: "<<PRINT_COLOR_RESET<<__O__<<std::endl
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

void loadFeatures(vector<vector<cv::Mat>> &features, const vector<string> &image_paths);

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
      if (ext.empty() || (!ext.empty() && file_ext == ext)) {
        files.emplace_back(i->path().string());
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
DEFINE_string(dir, "", "directory to use");
DEFINE_string(ext, ".jpg", "extension to look for");
DEFINE_int32(branching_level, 9, "internal: k-d tree branch level");
DEFINE_int32(depth_factors, 3, "internal: k-d tree depth factor");
DEFINE_string(save_db, "_db.yml.gz", "output postfix db name");
DEFINE_string(save_voc, "_voc.yml.gz", "output postfix voc name");

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

static bool fileExists(const std::string &path) {
  struct stat buffer = {0};
  return (stat(path.c_str(), &buffer) == 0);
}


DEFINE_validator(dir, &ValidatePathIsDirectory);

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// main
/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  auto image_paths = listFiles(FLAGS_dir, FLAGS_ext);
  if(image_paths.empty()) {
    RR("Found 0 files with extension " << FLAGS_ext << " in " << FLAGS_dir);
    exit(EXIT_FAILURE);
  }
  EM("Found " << image_paths.size() << " " << FLAGS_ext << " files");
  int preview_size = static_cast<int>(image_paths.size() > 10 ? 10 : image_paths.size());
  EM("Sample " << preview_size << " files:");
  for (int i = 0; i < preview_size; i++) {
    EM(image_paths[i]);
  }

  const path &stem = path(FLAGS_dir).stem();
  string folder_name(stem.string());
  FLAGS_save_voc =
      folder_name + "_" + to_string(FLAGS_branching_level) + "-" + to_string(FLAGS_depth_factors) + FLAGS_save_voc;
  FLAGS_save_db =
      folder_name + "_" + to_string(FLAGS_branching_level) + "-" + to_string(FLAGS_depth_factors) + FLAGS_save_db;

  if(fileExists(FLAGS_save_db)) {
    WN("File exists: " << FLAGS_save_db);
    WN("It will be over-written!");
    WN("Proceed?");
    wait();
  }
  if(fileExists(FLAGS_save_voc)) {
    WN("File exists: " << FLAGS_save_voc);
    WN("It will be over-written!");
    WN("Proceed?");
    wait();
  }

  EM("Using folder: " << folder_name);
  EM("-dir            = " << FLAGS_dir);
  EM("-ext            = " << FLAGS_ext);
  EM("-save_voc       = " << FLAGS_save_voc);
  EM("-save_db        = " << FLAGS_save_db);
  EM("-depth_factors  = " << FLAGS_depth_factors);
  EM("-branching_level= " << FLAGS_branching_level);

  wait();

  vector<vector<cv::Mat> > features;
  loadFeatures(features, image_paths);

  testVocCreation(features, image_paths);

  wait();

  testDatabase(features, image_paths);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat>> &features, const vector<string> &image_paths){
  features.clear();

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (int i = 0; i < image_paths.size(); i++) {
    DB(i << "/" << image_paths.size() << " imread " << image_paths[i]);
    cv::Mat image = cv::imread(image_paths[i], 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.emplace_back();
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
  const WeightingType weight = TF_IDF; // TF_IDF, TF, IDF, BINARY
  const ScoringType score = L1_NORM; // L1_NORM, L2_NORM, CHI_SQUARE, KL, BHATTACHARYYA, DOT_PRODUCT

  OrbVocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  int img_size = static_cast<int>(image_paths.size());
  for (int i = 0; i < img_size; i++) {
    voc.transform(features[i], v1);
    for (int j = 0; j < img_size; j++) {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary... " << FLAGS_save_voc<< endl;
  voc.save(FLAGS_save_voc);
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(vector<vector<cv::Mat>> features, vector<string> image_paths) {
  cout << "Creating a small database..." << endl;
  int img_size = static_cast<int>(image_paths.size());

  // load the vocabulary from disk
  OrbVocabulary voc(FLAGS_save_voc);

  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (int i = 0; i < img_size; i++) {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for (int i = 0; i < img_size; i++) {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database... " << FLAGS_save_db << endl;
  db.save(FLAGS_save_db);
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2(FLAGS_save_db);
  cout << "... done! This is: " << FLAGS_save_db << endl << db2 << endl;
}

// ----------------------------------------------------------------------------
