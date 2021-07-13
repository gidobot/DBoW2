/**
 * File: make_sift_vocabulary.cpp
 * Date: July 2021
 * Author: Gideon Billings
 * Description: application to create SIFT vocabulary for DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <cmath>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace DBoW2;
using namespace std;


// ----------------------------------------------------------------------------

static void ShowUsage(const std::string& name) {
  std::cerr << "Usage: " << name << " file" << std::endl;
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  // Parse arguments
  std::string base_dir = "";
  if (argc != 1 && argc != 2) {
    ShowUsage(argv[0]);
    return 0;
  }
  if (argc == 2) {
    base_dir = argv[1];
  } else {
    ShowUsage(argv[0]);
    return 0;
  }

  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  // load vocabulary
  cout << "Retrieving yaml vocabulary..." << endl;
  try {
    OrbVocabulary voc(k, L, weight, scoring);
    voc.loadFromTextFile(base_dir);
    cout << "... done! Vocabulary info: " << endl << voc << endl;
  }
  catch (const std::string &exc)
  {
    std::cerr << exc << std::endl;
  }

  return 0;
}
