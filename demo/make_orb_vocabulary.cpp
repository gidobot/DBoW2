/**
 * File: make_orb_vocabulary.cpp
 * Date: July 2021
 * Author: Gideon Billings
 * Description: application to create ORB vocabulary for DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <cmath>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace DBoW2;
using namespace std;
namespace fs = boost::filesystem;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features, const std::string img_dir);

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);

void vocCreation(const vector<vector<cv::Mat> > &features, const std::string base_dir);

// ----------------------------------------------------------------------------

static void ShowUsage(const std::string& name) {
  std::cerr << "Usage: " << name << " image_directory" << std::endl;
}

template <typename T>
T vectorNorm(std::vector<T> const& vec) {
  T val = 0;
  for (int i=0; i<vec.size(); i++)
    val += vec[i]*vec[i];
  val = sqrt(val);
  return val;
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  // Parse arguments
  std::string img_dir = "";
  if (argc != 1 && argc != 2) {
    ShowUsage(argv[0]);
    return 0;
  }
  if (argc == 2) {
    img_dir = argv[1];
  } else {
    ShowUsage(argv[0]);
    return 0;
  }

  vector<vector<cv::Mat > > features;
  loadFeatures(features, img_dir);

  vocCreation(features, img_dir);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, const std::string img_dir)
{
  features.clear();

   cv::Ptr<cv::ORB> orb = cv::ORB::create();

  // Sort directory of images
  typedef std::vector<fs::path> path_vec;
  path_vec v;
  copy(fs::recursive_directory_iterator(img_dir), fs::recursive_directory_iterator(),
    back_inserter(v));
  path_vec::const_iterator it(v.begin());
  // advance(it, 6757);

  int nimgs = v.size();
  features.reserve(nimgs);

  // Loop over the images
  cout << "Extracting ORB features..." << endl;
  int i = 1;
  while (it != v.end()) {
    cout << i << " of " << nimgs << std::endl;
    i++;
    // Check if the directory entry is a directory.
    if (fs::is_directory(*it)) {
      it++;
      continue;
    }
    // Check extension is correct
    if (it->extension() != ".png" && it->extension() != ".PNG") {
      it++;
      continue;
    }

    // Open the image
    std::string filename = it->string();
    cout << filename << std::endl;
    // Check image is loaded
    cv::Mat img = cv::imread(filename, 0); // CUDA SIFT expects greyscale images
    // cout << img.isContinuous() << std::endl;
    // cout << img << std::endl;
    if ( !(img.rows > 0 && img.cols > 0) ) {
      it++;
      continue;
    }
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // std::string windowName = "image";
    // cv::namedWindow(windowName);
    // cv::imshow(windowName, img);
    // cv::waitKey(0);
    // cv::destroyWindow(windowName);

    orb->detectAndCompute(img, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());

    it++;
  }

}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void vocCreation(const vector<vector<cv::Mat > > &features, const std::string base_dir)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save(base_dir + "/" + std::string("orb_voc.yml.gz"));
  cout << "Done" << endl;
}
