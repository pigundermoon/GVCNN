#include "caffe_head.h"
#include <time.h>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\gpu\gpumat.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define USE_OPENCV 1
//#define CPU_ONLY 1

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using GpuMat = cv::gpu::GpuMat;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Interpolator {
public:
	Interpolator(const string& model_file,
		const string& trained_file, bool ifGPU = false);

	void changemodel(const string& trained_file);
	std::vector<cv::Mat>  interpolate(const cv::Mat& img);
	std::vector<cv::Mat>  interpolate_4x(const cv::Mat& img);
	std::vector<cv::Mat>  interpolate_4x_hq(const cv::Mat& img);
	std::vector<cv::Mat>  interpolate_4x_1(const cv::Mat& img);

	void interpolate_test();

private:

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void WrapOutputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

	void WrapInputLayer_GPU(std::vector<GpuMat>* input_channels);

	void WrapOutputLayer_GPU(std::vector<GpuMat>* input_channels);

	void Preprocess_GPU(const cv::Mat& img,
		std::vector<GpuMat>* input_channels);

private:
	std::shared_ptr<Net<float> > net_;
	boost::shared_ptr<Net<float> > net;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
	int basesize;
};