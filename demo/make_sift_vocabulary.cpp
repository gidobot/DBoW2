/**
 * File: make_sift_vocabulary.cpp
 * Date: July 2021
 * Author: Gideon Billings
 * Description: application to create SIFT vocabulary for DBoW2
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

#include "cudaImage.h"
#include "cudaSift.h"

using namespace DBoW2;
using namespace std;
namespace fs = boost::filesystem;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<vector<float> > > &features, const std::string img_dir);

void changeStructure(const cv::Mat &plain, vector<vector<float>> &out);

void vocCreation(const vector<vector<vector<float> > > &features, const std::string base_dir);

void computeFeatures(const cv::Mat& I, SiftData &siftdata);

void copyCUDAFeatures(cv::Mat &descriptors, SiftData &data);

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

  vector<vector<vector<float> > > features;
  loadFeatures(features, img_dir);

  vocCreation(features, img_dir);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features, const std::string img_dir)
{
  features.clear();

  // Create CUDA based SIFT extractor
  SiftData sdata;
  InitSiftData(sdata, 2000, true, true);

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
  cout << "Extracting SIFT features..." << endl;
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

    // std::string windowName = "image";
    // cv::namedWindow(windowName);
    // cv::imshow(windowName, img);
    // cv::waitKey(0);
    // cv::destroyWindow(windowName);

    // Extract SIFT on GPU
    computeFeatures(img, sdata);

    // Copy features to cv::Mat
    cv::Mat descriptors;
    copyCUDAFeatures(descriptors, sdata);

    features.push_back(vector<vector<float> >());
    changeStructure(descriptors, features.back());

    it++;
  }

  FreeSiftData(sdata);

}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<vector<float>> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i].resize(128);
    memcpy(out[i].data(), plain.row(i).data, 128*sizeof(float));
  }
}

// ----------------------------------------------------------------------------

void vocCreation(const vector<vector<vector<float> > > &features, const std::string base_dir)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  SiftVocabulary voc(k, L, weight, scoring);

  cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save(base_dir + "/" + std::string("sift_voc.yml.gz"));
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void copyCUDAFeatures(cv::Mat &descriptors, SiftData &data) {
  #ifdef MANAGEDMEM
    SiftPoint *sift = data.m_data;
  #else
    SiftPoint *sift = data.h_data;
  #endif

  descriptors = cv::Mat(data.numPts, 128, CV_32F);

  for (int32_t i=0; i<data.numPts; i++) {
    std::memcpy(descriptors.row(i).data, sift[i].data, 128*sizeof(float));
  }
}

// ----------------------------------------------------------------------------

void computeFeatures(const cv::Mat& I, SiftData &siftdata) {
  cv::Mat gI;
  I.convertTo(gI, CV_32FC1);
  CudaImage img;
  img.Allocate(gI.cols, gI.rows, iAlignUp(gI.cols, 128), false, NULL, (float*)gI.data);
  img.Download();

  float initBlur = 1.0f;
  float thresh = 1.5f;

  // float *memoryTmpCUDA = AllocSiftTempMemory(I.cols, I.rows, 5, false);
  // ExtractSift(siftdata, img, 5, initBlur, thresh, 0.0f, false, memoryTmpCUDA);
  float *memoryTmpCUDA = AllocSiftTempMemory(I.cols*2, I.rows*2, 5, false);
  ExtractSift(siftdata, img, 5, initBlur, thresh, 0.0f, true, memoryTmpCUDA);
  FreeSiftTempMemory(memoryTmpCUDA);
}
