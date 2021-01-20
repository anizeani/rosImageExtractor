
// HEADER
#include "image-extraction/image-extraction.h"

// NON-SYSTEM
#include <fw-human-detection/falsecolor.h>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

namespace human_detection_pipeline {

ImageExtraction::ImageExtraction() : frame_idx_flir_(0u), frame_idx_rgb_(0u) {
  CHECK(loadSettings());
  // Create output directories
  if (!boost::filesystem::exists(output_directory_)) {
    LOG(INFO) << "Creating directory " << output_directory_;
    boost::filesystem::create_directory(output_directory_);
  }
  output_directory_rgb_ = output_directory_ + "/01_optical_raw/";
  output_directory_flir_ = output_directory_ + "/02_infrared_raw/";
  if (!boost::filesystem::exists(output_directory_rgb_)) {
    LOG(INFO) << "Creating directory " << output_directory_rgb_;
    boost::filesystem::create_directory(output_directory_rgb_);
  }
  if (!boost::filesystem::exists(output_directory_flir_)) {
    LOG(INFO) << "Creating directory " << output_directory_flir_;
    boost::filesystem::create_directory(output_directory_flir_);
  }
  fs_rgb_.open(output_directory_rgb_ + "timestamps.txt");
  fs_flir_.open(output_directory_flir_ + "timestamps.txt");
  fs_rgb_annotations_.open(output_directory_rgb_ + "annotations.txt");
  fs_flir_annotations_.open(output_directory_flir_ + "annotations.txt");

  std::ofstream fs_rgb_classes_rgb;
  fs_rgb_classes_rgb.open(output_directory_rgb_ + "classes.csv");
  LOG(INFO) << "Writing classes to " << output_directory_rgb_ + "classes.csv";
  fs_rgb_classes_rgb << "human, 0" << std::endl;
  fs_rgb_classes_rgb.close();

  std::ofstream fs_rgb_classes_flir;
  fs_rgb_classes_flir.open(output_directory_flir_ + "classes.csv");
  LOG(INFO) << "Writing classes to " << output_directory_flir_ + "classes.csv";
  fs_rgb_classes_flir << "human, 0" << std::endl;
  fs_rgb_classes_flir.close();
}

bool ImageExtraction::loadSettings() {
  // Load all settings.
  const std::string base = "/image_extraction/";
  io_handler_.loadRosParam(base + "rosbag_filename", &rosbag_filename_);
  io_handler_.loadRosParam(base + "output_directory", &output_directory_);

  io_handler_.loadRosParam(base + "flir/topic", &rosbag_topic_flir_);
  io_handler_.loadRosParam(base + "flir/rotate_code", &rotate_code_flir_);
  io_handler_.loadRosParam(base + "flir/extract", &extract_flir_);

  io_handler_.loadRosParam(base + "rgb/topic", &rosbag_topic_rgb_);
  io_handler_.loadRosParam(base + "rgb/rotate_code", &rotate_code_rgb_);
  io_handler_.loadRosParam(base + "rgb/extract", &extract_rgb_);

  io_handler_.loadRosParam(base + "show_images", &show_images_);

  // Image post-processing
  io_handler_.loadRosParam(base + "perform_alpha_beta", &perform_alpha_beta_);
  io_handler_.loadRosParam(base + "perform_gamma", &perform_gamma_);
  io_handler_.loadRosParam(base + "gamma", &gamma_);
  io_handler_.loadRosParam(base + "beta", &beta_);
  io_handler_.loadRosParam(base + "alpha", &alpha_);

  // Success.
  return true;
}

using namespace cv;

cv::Mat equalizeIntensity(const Mat& inputImage) {
  if (inputImage.channels() >= 3) {
    Mat ycrcb;

    cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

    std::vector<Mat> channels;
    split(ycrcb, channels);

    equalizeHist(channels[0], channels[0]);

    Mat result;
    merge(channels, ycrcb);

    cvtColor(ycrcb, result, CV_YCrCb2BGR);

    return result;
  }
  return Mat();
}

void ImageExtraction::extractImages() {
  try {
    bag_.reset(new rosbag::Bag);
    bag_->open(rosbag_filename_, rosbag::bagmode::Read);
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Could not open the rosbag " << rosbag_filename_ << ": "
               << ex.what();
  }
  LOG(INFO) << "Reading from rosbag: " << rosbag_filename_;

  std::vector<std::string> all_topics;
  all_topics.push_back(rosbag_topic_flir_);
  all_topics.push_back(rosbag_topic_rgb_);
  try {
    CHECK(bag_);
    bag_view_.reset(new rosbag::View(*bag_, rosbag::TopicQuery(all_topics)));
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Could not open a rosbag view: " << ex.what();
  }

  // Play all messages.
  CHECK(bag_view_);
  for (const rosbag::MessageInstance& message : *bag_view_) {
    // Enqueue image messages.
    sensor_msgs::ImageConstPtr image_message =
        message.instantiate<sensor_msgs::Image>();

    if (image_message) {
      CHECK(image_message);
      if (message.getTopic() == rosbag_topic_flir_ && extract_flir_) {
        // Get the image.
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
          cv_ptr = cv_bridge::toCvShare(image_message,
                                        sensor_msgs::image_encodings::MONO16);
        } catch (const cv_bridge::Exception& e) {
          LOG(FATAL) << "cv_bridge exception: " << e.what();
        }
        CHECK(cv_ptr);

        // Convert thermal image from 16bit to 8bit.
        bool do_temperature_scaling = false;
        cv::Mat thermal_image_16_bit = cv_ptr->image.clone();

        cv::Mat thermal_image_8_bit, thermal_image_8_bit_false_colored;
        thermal_image_8_bit_false_colored.create(
            thermal_image_16_bit.rows, thermal_image_16_bit.cols, CV_8UC1);
        converter_16_8::Instance().convert_to8bit(
            thermal_image_16_bit, thermal_image_8_bit, do_temperature_scaling);

        // Convert false color.
        bool temperature_legend = false;
        convertFalseColor(thermal_image_8_bit,
                          thermal_image_8_bit_false_colored,
                          palette::False_color_palette4, temperature_legend,
                          converter_16_8::Instance().getMin(),
                          converter_16_8::Instance().getMax());
        cv::rotate(thermal_image_8_bit_false_colored,
                   thermal_image_8_bit_false_colored, rotate_code_flir_);

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(4) << ++frame_idx_flir_;
        const std::string filename_flir_image =
            output_directory_flir_ + "flir_" + ss.str() + ".jpg";
        cv::imwrite(filename_flir_image, thermal_image_8_bit_false_colored);

        if (show_images_) {
          cv::imshow("thermal_image_8_bit_false_colored",
                     thermal_image_8_bit_false_colored);
          cv::waitKey(10);
        }

        fs_flir_ << filename_flir_image << " " << std::setprecision(15)
                 << image_message->header.stamp.toNSec() << " "
                 << image_message->header.seq << std::endl;
        fs_flir_.flush();

        fs_flir_annotations_ << filename_flir_image << ",,,,," << std::endl;
        fs_flir_annotations_.flush();
      }  // if flir
      if (message.getTopic() == rosbag_topic_rgb_ && extract_rgb_) {
        // Get the image.
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
          cv_ptr = cv_bridge::toCvShare(image_message,
                                        sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& e) {
          LOG(FATAL) << "cv_bridge exception: " << e.what();
        }
        CHECK(cv_ptr);

        cv::Mat image_original = cv_ptr->image.clone();
        cv::Mat image = cv_ptr->image.clone();

        /// Apply Histogram Equalization
        // image = equalizeIntensity(image);
        //        cv::Mat dst;
        //        cv::equalizeHist(image, dst );
        //        image = dst;

        if (perform_gamma_) {
          cv::Mat image_post_gamma;
          gammaCorrection(gamma_, image, &image_post_gamma);
          image = image_post_gamma;
        }

        if (perform_alpha_beta_) {
          cv::Mat image_post_alpha_beta;
          alphaBetaCorrection(alpha_, beta_, image, &image_post_alpha_beta);
          image = image_post_alpha_beta;
        }

        if (show_images_) {
          cv::Mat both, both_small;
          cv::hconcat(image_original, image, both);
          cv::resize(both, both_small, cv::Size(2 * 752, 480));
          cv::imshow("both", both_small);
          cv::waitKey(10);
        }

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(4) << ++frame_idx_rgb_;
        const std::string filename_rgb_image =
            output_directory_rgb_ + "rgb_" + ss.str() + ".jpg";
        cv::imwrite(filename_rgb_image, image);
        fs_rgb_ << filename_rgb_image << " " << std::setprecision(15)
                << image_message->header.stamp.toNSec() << " "
                << image_message->header.seq << std::endl;
        fs_rgb_.flush();

        fs_rgb_annotations_ << filename_rgb_image << ",,,,," << std::endl;
        fs_rgb_annotations_.flush();
      }
    }
  }
  LOG(INFO) << "Finished!";

  fs_rgb_.close();
  fs_rgb_annotations_.close();
  fs_flir_.close();
  fs_flir_annotations_.close();
}
}  // namespace human_detection_pipeline
