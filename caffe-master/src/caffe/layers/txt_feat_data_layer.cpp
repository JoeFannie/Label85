#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/txt_feat_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

	template <typename Dtype>
	TxtFeatDataLayer<Dtype>::~TxtFeatDataLayer<Dtype>() {
		this->StopInternalThread();
	}
	template <typename Dtype>
	int TxtFeatDataLayer<Dtype>::DisturbLabel(const int label, const float alpha, const int max_label)
	{
		if (alpha < 1e-6)
		{
			return label;
		}
		else
		{
			int inv_alpha = ceil(1.0f / alpha);
			int flag = caffe_rng_rand() % (inv_alpha + 2);
			//fprintf(stdout, "inv_alpha %d flag %d rand %d max_label %d\n", inv_alpha, flag, caffe_rng_rand(), max_label);
			if (flag > inv_alpha)
			{
				return caffe_rng_rand() % (max_label+1);
			}
			else{
				return label;
			}
		}

	}


	template <typename Dtype>
	void TxtFeatDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//string root_folder = this->layer_param_.txtfeat_data_param().root_folder();
		feat_dim_ = this->layer_param_.txtfeat_data_param().feat_dim();
		b_bin_file_ = this->layer_param_.txtfeat_data_param().b_binary_file();

		// Read the file with filenames and labels
		const string& source = this->layer_param_.txtfeat_data_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		string filename;
		FACE_IMAGE_INFO face_info;
		int label;
		max_label_ = 0;
		while (infile >> filename >> face_info.xy[0] >> face_info.xy[1] >> face_info.xy[2] >> face_info.xy[3] >> face_info.label) {
			lines_.push_back(std::make_pair(filename, face_info));
			max_label_ = std::max(max_label_, face_info.label);
		}

		dislab_alpha_ = this->layer_param_.txtfeat_data_param().dislab_alpha();
		CHECK_GE(dislab_alpha_, 0);

		if (this->layer_param_.txtfeat_data_param().shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleTxtFeats();
		}
		LOG(INFO) << "A total of " << lines_.size() << " images.";

		lines_id_ = 0;
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.txtfeat_data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.txtfeat_data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}


		// Use data_transformer to infer the expected blob shape from a cv_image.
		//vector<int> top_shape = this->data_transformer_->InferFaceBlobShape(cv_img); //返回一个4维向量， num, img_channels, width, height
		vector<int> top_shape(4);
		top_shape[0] = 1;
		top_shape[1] = feat_dim_;
		top_shape[2] = 1;
		top_shape[3] = 1;
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.txtfeat_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";

		data_buf_.reset(new Blob<float>(1, feat_dim_, 1, 1));

		top_shape[0] = batch_size;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		top[0]->Reshape(top_shape);

		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		vector<int> label_shape(1, batch_size);
		top[1]->Reshape(label_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(label_shape);
		}
	}

	template <typename Dtype>
	void TxtFeatDataLayer<Dtype>::ShuffleTxtFeats() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	template <typename Dtype>
	bool TxtFeatDataLayer<Dtype>::ReadBinTxtFeat(const string& filename,
		const int feat_dim, float *buffer) {

		std::ifstream infile(filename.c_str(), ios::in | ios::binary);
		if (!infile.is_open())
		{
			LOG(ERROR) << "failed  to open read file:" << filename.c_str() << std::endl;
			return false;
		}

		infile.read((char *)buffer, sizeof(float)*feat_dim);
		return true;
	}
	template <typename Dtype>
	bool TxtFeatDataLayer<Dtype>::ReadTxtFeat(const string& filename,
		const int feat_dim, float *buffer) {
		std::ifstream infile(filename.c_str());
		if (!infile.is_open())
		{
			LOG(ERROR) << "failed  to open read file:" << filename.c_str() << std::endl;
			return false;
		}

		std::stringstream ss;
		std::string str;
		std::getline(infile, str);
		ss.clear();
		ss.str(str);
		for (int i = 0; i < feat_dim; i++)
		{
			ss >> buffer[i];
		}

		return true;
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void TxtFeatDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		TxtFeatDataParameter txtfeat_data_param = this->layer_param_.txtfeat_data_param();
		const int batch_size = txtfeat_data_param.batch_size();
		string root_folder = txtfeat_data_param.root_folder();

		// Reshape according to the first image of each batch
		// on single input batches allows for inputs of varying dimension.
		vector<int> top_shape(4);
		top_shape[0] = 1;
		top_shape[1] = feat_dim_;
		top_shape[2] = 1;
		top_shape[3] = 1;

		//this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();
		
		// datum scales
		const int lines_size = lines_.size();

		//normal fetch data
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			CHECK_GT(lines_size, lines_id_);
			//cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, false);
			//CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply transformations (mirror, crop...) to the image
			int offset = batch->data_.offset(item_id);
			bool b_read;
			if (b_bin_file_)
			{
				b_read = ReadBinTxtFeat(root_folder + lines_[lines_id_].first, feat_dim_, data_buf_->mutable_cpu_data());
			}
			else
			{
				b_read = ReadTxtFeat(root_folder + lines_[lines_id_].first, feat_dim_, data_buf_->mutable_cpu_data());
			}
			
			const float *data_buf = data_buf_->cpu_data();
			for (int i = 0; i < feat_dim_; i++)
			{
				prefetch_data[offset + i] = data_buf[i];
			}
			trans_time += timer.MicroSeconds();

			//prefetch_label[item_id] = lines_[lines_id_].second.label;

			prefetch_label[item_id] = DisturbLabel(lines_[lines_id_].second.label, dislab_alpha_, max_label_);
			//fprintf(stdout, "%f  %d  %f  %d\n", prefetch_label[item_id], lines_[lines_id_].second.label, dislab_alpha_, max_label_);
			// go to the next iter
			lines_id_++;
			if (lines_id_ >= lines_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
				if (this->layer_param_.txtfeat_data_param().shuffle()) {
					ShuffleTxtFeats();
				}
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

	}

	INSTANTIATE_CLASS(TxtFeatDataLayer);
	REGISTER_LAYER_CLASS(TxtFeatData);

}  // namespace caffe
#endif  // USE_OPENCV
