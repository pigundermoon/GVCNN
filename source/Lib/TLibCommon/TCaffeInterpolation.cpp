// CaffeTest.cpp : Defines the entry point for the console application.
//

#include "TCaffeInterpolation.h"


//Interpolator::Interpolator(const string& model_file, const string& trained_file, bool ifGPU) {
//
//
//	if (ifGPU)
//	{
//		Caffe::set_mode(Caffe::GPU);
//	}
//	else
//	{
//		Caffe::set_mode(Caffe::CPU);
//	}
//	net.reset(new Net<float>(model_file, TEST));
//	net->CopyTrainedLayersFrom(trained_file);
//
//}

//std::vector<cv::Mat> Interpolator::interpolate(const cv::Mat& img) {
//	
//	cv::Mat sample;
//	if (img.channels() == 3)
//	{
//		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//	}
//	else if (img.channels() == 4)
//	{
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//		cv::cvtColor(sample, sample, cv::COLOR_BGR2GRAY);
//	}
//	else if (img.channels() == 1)
//	{
//		sample = img;
//	}
//
//	cv::Mat sample_resized;
//	sample_resized = sample;
//
//	cv::Mat sample_float;
//	sample_resized.convertTo(sample_float, CV_32FC1);
//
//	sample_float = sample_float / 255.0;
//
//	int batchsize = 1;
//	boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net->layer_by_name("data"));
//	Blob<float> datashape(batchsize, 1, sample_float.rows, sample_float.cols);
//	net->top_vecs()[0].at(0)->Reshape(datashape.shape());
//	net->Reshape();
//
//	vector<int> labels = { 0 };
//	std::vector<cv::Mat> imgs = { sample_float };
//
//	memory_data_layer ->AddMatVector(imgs, labels);
//
//    vector<Blob<float>*> result = net->Forward();
//
//	int channels = result[0]->shape(1), height = result[0]->shape(2), width = result[0]->shape(3);
//	float cur;
//	const float* data_ptr = (const float*)result[0]->cpu_data();
//	std::vector<cv::Mat> results;
//	cv::Mat_<float> image(height, width, CV_32FC1);
//	for (int l = 0; l < channels; l++) 
//	{
//		for (int i = 0; i < height; i++) 
//		{
//			for (int j = 0; j < width; j++) 
//			{
//				cur = *(data_ptr + l * width * height + i * width + j);
//				image(i,j)= cur;
//			}
//		}
//		results.push_back(image);
//	}
//
//	cv::Mat tmp0 = results[0];
//	cv::Mat tmp1 = results[1];
//	cv::Mat tmp2 = results[2];
//
//	return results;
//}
//
//std::vector<cv::Mat> Interpolator::interpolate_4x(const cv::Mat& img) {
//
//	Blob<float>* input_layer = net_->input_blobs()[0];
//
//	input_geometry_ = cv::Size(img.cols, img.rows);
//
//	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
//	net_->Reshape();
//
//	std::vector<cv::Mat> input_channels;
//	WrapInputLayer(&input_channels);
//
//	Blob<float>* output_layer = net_->output_blobs()[0];
//	std::vector<cv::Mat> output_channels;
//	WrapOutputLayer(&output_channels);
//
//	Preprocess(img, &input_channels);
//
//	net_->Forward();
//
//	std::vector<cv::Mat> result;
//
//	result.push_back(output_channels[0]);
//	result.push_back(output_channels[1]);
//	result.push_back(output_channels[2]);
//	result.push_back(output_channels[3]);
//	result.push_back(output_channels[4]);
//	result.push_back(output_channels[5]);
//	result.push_back(output_channels[6]);
//	result.push_back(output_channels[7]);
//	result.push_back(output_channels[8]);
//	result.push_back(output_channels[9]);
//	result.push_back(output_channels[10]);
//	result.push_back(output_channels[11]);
//	result.push_back(output_channels[12]);
//	result.push_back(output_channels[13]);
//	result.push_back(output_channels[14]);
//
//	cv::Mat tmp = result[0];
//
//	return result;
//}

void Interpolator::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data;

	if (Caffe::mode() == Caffe::CPU)
	{
		input_data = input_layer->mutable_cpu_data();
	}
	else
	{
		//input_data = input_layer->mutable_cpu_data();
		input_data = input_layer->mutable_gpu_data();
	}
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		channel.at<float>(1, 1) = 1;
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Interpolator::WrapInputLayer_GPU(std::vector<GpuMat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_gpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		GpuMat  channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Interpolator::WrapOutputLayer(std::vector<cv::Mat>* output_channels) {
	Blob<float>* ourput_layer = net_->output_blobs()[0];

	int width = ourput_layer->width();
	int height = ourput_layer->height();

	float* ouput_data;
	if (Caffe::mode() == Caffe::CPU)
	{
		ouput_data = ourput_layer->mutable_cpu_data();
	}
	else
	{
		//ouput_data = ourput_layer->mutable_cpu_data();
		ouput_data = ourput_layer->mutable_gpu_data();
	}
	for (int i = 0; i < ourput_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, ouput_data);
		output_channels->push_back(channel);
		ouput_data += width * height;
	}
}

void Interpolator::WrapOutputLayer_GPU(std::vector<GpuMat>* output_channels) {
	Blob<float>* ourput_layer = net_->output_blobs()[0];

	int width = ourput_layer->width();
	int height = ourput_layer->height();

	float* ouput_data = ourput_layer->mutable_gpu_data();

	for (int i = 0; i < ourput_layer->channels(); ++i) {
		GpuMat channel(height, width, CV_32FC1, ouput_data);
		output_channels->push_back(channel);
		ouput_data += width * height;
	}
}

void Interpolator::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {

	cv::Mat sample;
	if (img.channels() == 3)
	{
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	}
	else if (img.channels() == 4)
	{
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		cv::cvtColor(sample, sample, cv::COLOR_BGR2GRAY);
	}
	else if (img.channels() == 1)
	{
		sample = img;
	}

	cv::Mat sample_resized;
	sample_resized = sample;

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC1);

	sample_float = sample_float / 255.0;
	//sample_float = sample_float;
	cv::split(sample_float, *input_channels);

	if (Caffe::mode() == Caffe::CPU)
	{
		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}
	else
	{
		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->gpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}
	//sample_float.copyTo((*input_channels).at(0));

}

void Interpolator::Preprocess_GPU(const cv::Mat& img,
	std::vector<GpuMat>* input_channels) {

	cv::Mat sample_resized;
	sample_resized = img;

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC1);

	sample_float = sample_float / 255.0;

	GpuMat sample;
	sample.upload(sample_float);
	sample.copyTo((*input_channels).at(0));
	//(*input_channels).at(0) = sample;

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->gpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
	//sample_float.copyTo((*input_channels).at(0));

}

//
//int main(int argc, char** argv) 
//{
//
//	::google::InitGoogleLogging(argv[0]);
//
//	string model_file = "fast_interp_deploy_2x_l20.prototxt";
//	string trained_file = "fast_interp_2x_l20_iter_898500.caffemodel";
//
//	clock_t start_time = clock();
//	Interpolator interpolator(model_file, trained_file);
//	clock_t end_time = clock();
//
//	printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);
//	string file = "timg.jpg";
//	std::cout << "---------- Prediction for " << file << " ----------" << std::endl;
//	cv::Mat img = cv::imread(file, -1);
//	CHECK(!img.empty()) << "Unable to decode image " << file;
//	std::vector<cv::Mat>  predictions = interpolator.interpolate(img);
//	
//}

Interpolator::Interpolator(const string& model_file, const string& trained_file, bool ifGPU) {

	//ifGPU = false;
	if (ifGPU)
	{
		Caffe::set_mode(Caffe::GPU);
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
	}


	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	basesize = 200;

}

void Interpolator::changemodel(const string& trained_file)
{
	net_->CopyTrainedLayersFrom(trained_file);
}

void Interpolator::interpolate_test() {
	cv::Mat_<uchar> img = cv::Mat(400, 400, CV_8UC1);
	for (int i = 0; i < 400; i++)
	{
		for (int j = 0; j < 400; j++)
		{
			img(i, j) = i % 4 * 4 + j % 4 + 1;
		}
	}

	Blob<float>* input_layer = net_->input_blobs()[0];

	input_geometry_ = cv::Size(img.cols, img.rows);

	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Blob<float>* output_layer = net_->output_blobs()[0];
	std::vector<cv::Mat> output_channels;
	WrapOutputLayer(&output_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	std::vector<cv::Mat> result;

	result.push_back(output_channels[0]);
	result.push_back(output_channels[1]);
	result.push_back(output_channels[2]);
	result.push_back(output_channels[3]);
	result.push_back(output_channels[4]);
	result.push_back(output_channels[5]);
	result.push_back(output_channels[6]);
	result.push_back(output_channels[7]);
	result.push_back(output_channels[8]);
	result.push_back(output_channels[9]);
	result.push_back(output_channels[10]);
	result.push_back(output_channels[11]);
	result.push_back(output_channels[12]);
	result.push_back(output_channels[13]);
	result.push_back(output_channels[14]);
	result.push_back(output_channels[15]);

	cv::Mat tmp0 = result[0];
	cv::Mat tmp1 = result[1];
	cv::Mat tmp2 = result[2];
	cv::Mat tmp3 = result[3];
	cv::Mat tmp4 = result[4];
	cv::Mat tmp5 = result[5];
	cv::Mat tmp6 = result[6];
	cv::Mat tmp7 = result[7];
	cv::Mat tmp8 = result[8];
	cv::Mat tmp9 = result[9];
	cv::Mat tmp10 = result[10];
	cv::Mat tmp11 = result[11];
	cv::Mat tmp12 = result[12];
	cv::Mat tmp13 = result[13];
	cv::Mat tmp14 = result[14];
	cv::Mat tmp15 = result[15];

}

std::vector<cv::Mat> Interpolator::interpolate(const cv::Mat& img) {

	//Blob<float>* input_layer = net_->input_blobs()[0];

	//input_geometry_ = cv::Size(img.cols, img.rows);

	//input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	//net_->Reshape();

	//std::vector<cv::Mat> input_channels;
	//WrapInputLayer(&input_channels);

	//Blob<float>* output_layer = net_->output_blobs()[0];
	//std::vector<cv::Mat> output_channels;
	//WrapOutputLayer(&output_channels);

	//Preprocess(img, &input_channels);

	//net_->Forward();

	//std::vector<cv::Mat> result;

	//result.push_back(output_channels[0]);
	//result.push_back(output_channels[1]);
	//result.push_back(output_channels[2]);

	//return result;

	cv::Mat_<float> tmpsav0 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav1 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav2 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);

	Blob<float>* input_layer = net_->input_blobs()[0];

	for (int i = 0; i*basesize < img.rows; i++)
	{
		for (int j = 0; j*basesize < img.cols; j++)
		{
			//std::cout << "half i: " << i << " j: " << j << std::endl;
			int sti, stj, edi, edj;
			if (i == 0) sti = 0;
			else sti = i*basesize - 30;
			if (j == 0) stj = 0;
			else stj = j*basesize - 30;
			if ((i + 1)*basesize + 30 >= img.rows) edi = img.rows;
			else edi = (i + 1)*basesize + 30;
			if ((j + 1)*basesize + 30 >= img.cols) edj = img.cols;
			else edj = (j + 1)*basesize + 30;

			cv::Mat_<uchar> tmpimg = cv::Mat(edi - sti, edj - stj, CV_8UC1);
			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpimg(i - sti, j - stj) = img.at<uchar>(i, j);
				}
			}

			input_geometry_ = cv::Size(edj - stj, edi - sti );
			input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
			net_->Reshape();

			std::vector<cv::Mat> resmats;

			if (Caffe::mode() == Caffe::GPU)
			{
				std::vector<GpuMat> input_channels;
				WrapInputLayer_GPU(&input_channels);
				Blob<float>* output_layer = net_->output_blobs()[0];
				std::vector<GpuMat> output_channels;
				WrapOutputLayer_GPU(&output_channels);
				Preprocess_GPU(tmpimg, &input_channels);
				net_->Forward();
				for (int i = 0; i < 3; i++)
				{
					cv::Mat tmp;
					output_channels[i].download(tmp);
					resmats.push_back(tmp);
				}
			}
			else
			{
				std::vector<cv::Mat> input_channels;
				WrapInputLayer(&input_channels);

				Blob<float>* output_layer = net_->output_blobs()[0];
				std::vector<cv::Mat> output_channels;
				WrapOutputLayer(&output_channels);
				Preprocess(tmpimg, &input_channels);
				net_->Forward();
				for (int i = 0; i < 3; i++)
				{
					cv::Mat tmp;
					output_channels[i].copyTo(tmp);
					resmats.push_back(tmp);
				}
			}

			int di = 0;
			int dj = 0;
			if (sti != 0)
			{
				sti += 30;
				di = 30;
			}

			if (stj != 0)
			{
				stj += 30;
				dj = 30;
			}
			if (edi != img.rows) edi -= 30;
			if (edj != img.cols) edj -= 30;

			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpsav0(i, j) = resmats[0].at<float>(i - sti + di, j - stj + dj) /** 10 + 0.5*/;
					tmpsav1(i, j) = resmats[1].at<float>(i - sti + di, j - stj + dj) /** 10 + 0.5*/;
					tmpsav2(i, j) = resmats[2].at<float>(i - sti + di, j - stj + dj) /** 10 + 0.5*/;
				}
			}

		}
	}

	//cv::imwrite("hresidual1.jpg",tmpsav0*255);
	//cv::imwrite("hresidual2.jpg",tmpsav1*255);
	//cv::imwrite("hresidual3.jpg", tmpsav2*255);

	std::vector<cv::Mat> result;
	result.push_back(tmpsav0);
	result.push_back(tmpsav1);
	result.push_back(tmpsav2);

	return result;
}
std::vector<cv::Mat> Interpolator::interpolate_4x(const cv::Mat& img) {

	cv::Mat_<float> tmpsav0 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav1 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav2 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav3 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav4 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav5 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav6 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav7 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav8 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav9 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav10 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav11 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav12 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav13 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);
	cv::Mat_<float> tmpsav14 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);

	Blob<float>* input_layer = net_->input_blobs()[0];

	for (int i = 0; i*basesize < img.rows; i++)
	{
		for (int j = 0; j*basesize < img.cols; j++)
		{
			//std::cout << "quater i: " << i << " j: " << j << std::endl;
			int sti, stj, edi, edj;
			if (i == 0) sti = 0;
			else sti = i*basesize - 30;
			if (j == 0) stj = 0;
			else stj = j*basesize - 30;
			if ((i + 1)*basesize + 30 >= img.rows) edi = img.rows;
			else edi = (i + 1)*basesize + 30;
			if ((j + 1)*basesize + 30 >= img.cols) edj = img.cols;
			else edj = (j + 1)*basesize + 30;

			cv::Mat_<uchar> tmpimg = cv::Mat(edi - sti, edj - stj, CV_8UC1);
			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpimg(i - sti, j - stj) = img.at<uchar>(i, j);
				}
			}

			input_geometry_ = cv::Size(edj - stj, edi - sti);
			input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
			net_->Reshape();

			std::vector<cv::Mat> resmats;

			if (Caffe::mode() == Caffe::GPU)
			{
				std::vector<GpuMat> input_channels;
				WrapInputLayer_GPU(&input_channels);
				Blob<float>* output_layer = net_->output_blobs()[0];
				std::vector<GpuMat> output_channels;
				WrapOutputLayer_GPU(&output_channels);
				Preprocess_GPU(tmpimg, &input_channels);
				net_->Forward();
				for (int i = 0; i < 15; i++)
				{
					cv::Mat tmp;
					output_channels[i].download(tmp);
					resmats.push_back(tmp);
				}
			}
			else
			{
				std::vector<cv::Mat> input_channels;
				WrapInputLayer(&input_channels);

				Blob<float>* output_layer = net_->output_blobs()[0];
				std::vector<cv::Mat> output_channels;
				WrapOutputLayer(&output_channels);
				Preprocess(tmpimg, &input_channels);
				net_->Forward();
				for (int i = 0; i < 15; i++)
				{
					cv::Mat tmp;
					output_channels[i].copyTo(tmp);
					resmats.push_back(tmp);
				}
			}

			int di = 0;
			int dj = 0;
			if (sti != 0)
			{
				sti += 30;
				di = 30;
			}

			if (stj != 0)
			{
				stj += 30;
				dj = 30;
			}
			if (edi != img.rows) edi -= 30;
			if (edj != img.cols) edj -= 30;

			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpsav0(i, j) = resmats[0].at<float>(i - sti + di, j - stj + dj)/**10 + 0.5*/;
					tmpsav1(i, j) = resmats[1].at<float>(i - sti + di, j - stj + dj);
					tmpsav2(i, j) = resmats[2].at<float>(i - sti + di, j - stj + dj);
					tmpsav3(i, j) = resmats[3].at<float>(i - sti + di, j - stj + dj);
					tmpsav4(i, j) = resmats[4].at<float>(i - sti + di, j - stj + dj);
					tmpsav5(i, j) = resmats[5].at<float>(i - sti + di, j - stj + dj);
					tmpsav6(i, j) = resmats[6].at<float>(i - sti + di, j - stj + dj);
					tmpsav7(i, j) = resmats[7].at<float>(i - sti + di, j - stj + dj);
					tmpsav8(i, j) = resmats[8].at<float>(i - sti + di, j - stj + dj);
					tmpsav9(i, j) = resmats[9].at<float>(i - sti + di, j - stj + dj);
					tmpsav10(i, j) = resmats[10].at<float>(i - sti + di, j - stj + dj);
					tmpsav11(i, j) = resmats[11].at<float>(i - sti + di, j - stj + dj);
					tmpsav12(i, j) = resmats[12].at<float>(i - sti + di, j - stj + dj);
					tmpsav13(i, j) = resmats[13].at<float>(i - sti + di, j - stj + dj);
					tmpsav14(i, j) = resmats[14].at<float>(i - sti + di, j - stj + dj)/**10 + 0.5*/;
				}
			}

		}
	}

	//cv::imwrite("residual1.jpg",tmpsav0*255);
	//cv::imwrite("residual12.jpg", tmpsav14*255);
	std::vector<cv::Mat> result;
	result.push_back(tmpsav0);
	result.push_back(tmpsav1);
	result.push_back(tmpsav2);
	result.push_back(tmpsav3);
	result.push_back(tmpsav4);
	result.push_back(tmpsav5);
	result.push_back(tmpsav6);
	result.push_back(tmpsav7);
	result.push_back(tmpsav8);
	result.push_back(tmpsav9);
	result.push_back(tmpsav10);
	result.push_back(tmpsav11);
	result.push_back(tmpsav12);
	result.push_back(tmpsav13);
	result.push_back(tmpsav14);

	return result;
}

std::vector<cv::Mat> Interpolator::interpolate_4x_hq(const cv::Mat& img) {

	Blob<float>* input_layer = net_->input_blobs()[0];

	input_geometry_ = cv::Size(img.cols, img.rows);

	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Blob<float>* output_layer = net_->output_blobs()[0];
	std::vector<cv::Mat> output_channels;
	WrapOutputLayer(&output_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	std::vector<cv::Mat> result;

	result.push_back(output_channels[0]);
	result.push_back(output_channels[1]);
	result.push_back(output_channels[2]);
	result.push_back(output_channels[3]);
	result.push_back(output_channels[4]);
	result.push_back(output_channels[5]);
	result.push_back(output_channels[6]);
	result.push_back(output_channels[7]);
	result.push_back(output_channels[8]);
	result.push_back(output_channels[9]);
	result.push_back(output_channels[10]);
	result.push_back(output_channels[11]);

	//cv::Mat tmp = result[0];
	//cv::Mat tmp1 = result[1];

	return result;
}

std::vector<cv::Mat> Interpolator::interpolate_4x_1(const cv::Mat& img) {

	//Blob<float>* input_layer = net_->input_blobs()[0];

	//input_geometry_ = cv::Size(img.cols, img.rows);

	//input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	//net_->Reshape();

	//std::vector<cv::Mat> input_channels;
	//WrapInputLayer(&input_channels);

	//Blob<float>* output_layer = net_->output_blobs()[0];
	//std::vector<cv::Mat> output_channels;
	//WrapOutputLayer(&output_channels);

	//Preprocess(img, &input_channels);

	//net_->Forward();

	//std::vector<cv::Mat> result;

	//result.push_back(output_channels[0]);

	////cv::Mat tmp = result[0];
	////cv::Mat tmp1 = result[1];

	//return result;

	cv::Mat_<float> tmpsav0 = cv::Mat_<float>(img.rows, img.cols, CV_32FC1);

	Blob<float>* input_layer = net_->input_blobs()[0];

	for (int i = 0; i*basesize < img.rows; i++)
	{
		for (int j = 0; j*basesize < img.cols; j++)
		{
			std::cout << "quater i: " << i << " j: " << j << std::endl;
			int sti, stj, edi, edj;
			if (i == 0) sti = 0;
			else sti = i*basesize - 30;
			if (j == 0) stj = 0;
			else stj = j*basesize - 30;
			if ((i + 1)*basesize + 30 >= img.rows) edi = img.rows;
			else edi = (i + 1)*basesize + 30;
			if ((j + 1)*basesize + 30 >= img.cols) edj = img.cols;
			else edj = (j + 1)*basesize + 30;

			cv::Mat_<uchar> tmpimg = cv::Mat(edi - sti + 1, edj - stj + 1, CV_8UC1);
			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpimg(i - sti, j - stj) = img.at<uchar>(i, j);
				}
			}

			input_geometry_ = cv::Size(edj - stj + 1, edi - sti + 1);
			input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
			net_->Reshape();

			std::vector<cv::Mat> input_channels;
			WrapInputLayer(&input_channels);

			Blob<float>* output_layer = net_->output_blobs()[0];
			std::vector<cv::Mat> output_channels;
			WrapOutputLayer(&output_channels);

			Preprocess(tmpimg, &input_channels);

			net_->Forward();

			int di = 0;
			int dj = 0;
			if (sti != 0)
			{
				sti += 30;
				di = 30;
			}

			if (stj != 0)
			{
				stj += 30;
				dj = 30;
			}
			if (edi != img.rows) edi -= 30;
			if (edj != img.cols) edj -= 30;

			for (int i = sti; i < edi; i++)
			{
				for (int j = stj; j < edj; j++)
				{
					tmpsav0(i, j) = output_channels[0].at<float>(i - sti + di, j - stj + dj);
				}
			}

		}
	}

	std::vector<cv::Mat> result;
	result.push_back(tmpsav0);

	return result;
}