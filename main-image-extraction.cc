/*
 *    Filename: main.cc
 *  Created on: 2018
 *      Author: Timo Hinzmann, Tobias Stegemann
 *   Institute: ETH Zurich, Autonomous Systems Lab
 *     License: Apache License Version 2.0
 */

// SYSTEM
#include <iostream>
#include <string>

// NON-SYSTEM
#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <maplab-common/glog-helpers.h>
#include <ros/ros.h>

// PACKAGE
#include "image-extraction/image-extraction.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  google::InstallFailureFunction(&common::glogFailFunctionWithConsoleRecovery);

  // Set up ROS-node.
  ros::init(argc, argv, "fw_human_detection");
  human_detection_pipeline::ImageExtraction image_extraction;
  image_extraction.extractImages();
  ros::shutdown();

  return 0;
}
