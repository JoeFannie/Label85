#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/face_image_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	FaceImageLayer<Dtype>::~FaceImageLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void FaceImageLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		string root_folder = this->layer_param_.face_image_param().root_folder();

		// Read the file with filenames and labels
		const string& source = this->layer_param_.face_image_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		string filename;
		FACE_IMAGE_INFO face_info;
		int label;
		while (infile >> filename >> face_info.xy[0] >> face_info.xy[1] >> face_info.xy[2] >> face_info.xy[3] >> face_info.label) {
			lines_.push_back(std::make_pair(filename, face_info));
		}

		if (this->layer_param_.face_image_param().shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}
		LOG(INFO) << "A total of " << lines_.size() << " images.";

		lines_id_ = 0;
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.face_image_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.face_image_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}
		// Read an image, and use it to initialize the top blob.
		//fprintf(stdout, "lines_id_: %d size:%d\n", lines_id_, lines_.size());
		//fprintf(stdout, "source: %s\n", source.c_str());
		//fprintf(stdout, "file: %s\n", lines_[lines_id_].first.c_str());
		//for (int i = 0; i < lines_.size(); i++)
		//{
		//	fprintf(stdout, "%s %d %d %d %d %d\n", lines_[i].first.c_str(), lines_[i].second.xy[0], lines_[i].second.xy[1], lines_[i].second.xy[2], lines_[i].second.xy[3], lines_[i].second.label );
		//}

		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, false);
		CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		// Use data_transformer to infer the expected blob shape from a cv_image.
		vector<int> top_shape = this->data_transformer_->InferFaceBlobShape(cv_img); //返回一个4维向量， num, img_channels, width, height
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.face_image_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";

		if (this->layer_param_.face_image_param().triplet_ver())
		{
			CHECK_EQ(batch_size%3, 0) << "For triplet_ver signal, batch_size should divisible by 3";
		}

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
	void FaceImageLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}
	template <typename Dtype>
	int FaceImageLayer<Dtype>::find_np_sample(int anchor_idx, int anchor_label, bool find_pos)
	{
		const int lines_size = lines_.size();

		int curr_idx;
		int start = caffe_rng_rand() % lines_size;

		for (int i = 0; i < lines_size; i++)
		{
			curr_idx = (start + i) % lines_size;
			if (find_pos)
			{
				if (lines_[curr_idx].second.label == anchor_label && curr_idx != anchor_idx)
				{
					return curr_idx;
				}
			}
			else
			{
				if (lines_[curr_idx].second.label != anchor_label)
				{
					return curr_idx;
				}
			}
		}

		if (find_pos)
		{
			return anchor_idx;
		}
		else
		{
			return -1;
		}

	}

	template <typename Dtype>
	void FaceImageLayer<Dtype>::get_one_triplet(int curr_line_id, int *res)
	{
		const int lines_size = lines_.size();
		CHECK(res);
		CHECK_GT(lines_size, curr_line_id);
		
		const int anchor_label = lines_[curr_line_id].second.label;
		res[0] = curr_line_id;
		if (lines_[(curr_line_id + 1 )% lines_size].second.label == anchor_label)
		{
			res[1] = (curr_line_id + 1) % lines_size;
			//find negitive sample
			res[2] = find_np_sample(res[0], anchor_label, false);
		}
		else{
			res[2] = (curr_line_id + 1) % lines_size;
			//find positive sample
			res[1] = find_np_sample(res[0], anchor_label, true);
		}
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void FaceImageLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		FaceImageParameter face_image_param = this->layer_param_.face_image_param();
		const int batch_size = face_image_param.batch_size();
		string root_folder = face_image_param.root_folder();

		// Reshape according to the first image of each batch
		// on single input batches allows for inputs of varying dimension.
		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, false);
		CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		// Use data_transformer to infer the expected blob shape from a cv_img.
		vector<int> top_shape = this->data_transformer_->InferFaceBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();
		// datum scales
		const int lines_size = lines_.size();


		if (this->layer_param_.face_image_param().triplet_ver())
		{
			int idx_arr[3];
			//Triplet data fetch
			for (int item_id = 0; item_id < batch_size/3; ++item_id) {
				// get a blob
				timer.Start();
				CHECK_GT(lines_size, lines_id_);

				get_one_triplet(lines_id_, idx_arr);
				//fprintf(stdout, "index: %d %d %d\n", idx_arr[0], idx_arr[1], idx_arr[2]);
				//fprintf(stdout, "label: %d %d %d\n", lines_[idx_arr[0]].second.label, lines_[idx_arr[1]].second.label, lines_[idx_arr[2]].second.label);
				//fprintf(stdout, "\n");

				CHECK_EQ(lines_[idx_arr[0]].second.label, lines_[idx_arr[1]].second.label);
				CHECK_NE(lines_[idx_arr[0]].second.label, lines_[idx_arr[2]].second.label);

				
				for (int i = 0; i < 3; i++)
				{
					cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[idx_arr[i]].first, false);
					CHECK(cv_img.data) << "Could not load " << lines_[idx_arr[i]].first;
					read_time += timer.MicroSeconds();
					timer.Start();
					// Apply transformations (mirror, crop...) to the image
					int offset = batch->data_.offset(item_id + i*batch_size / 3);
					this->transformed_data_.set_cpu_data(prefetch_data + offset);
					this->data_transformer_->FaceImageTransform(cv_img, lines_[idx_arr[i]].second.xy, &(this->transformed_data_));
					trans_time += timer.MicroSeconds();

					prefetch_label[item_id + i*batch_size / 3] = lines_[idx_arr[i]].second.label;
				}

				// go to the next iter
				lines_id_++;
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					if (this->layer_param_.face_image_param().shuffle()) {
						ShuffleImages();
					}
				}
			}

			for (int i = 0; i < batch_size / 3; i++)
			{
				CHECK_EQ(prefetch_label[i], prefetch_label[i + batch_size / 3]);
				CHECK_NE(prefetch_label[i], prefetch_label[i + 2 * batch_size / 3]);
			}
			batch_timer.Stop();
			DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
			DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
			DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
		}
		else{
			//normal fetch data
			for (int item_id = 0; item_id < batch_size; ++item_id) {
				// get a blob
				timer.Start();
				CHECK_GT(lines_size, lines_id_);
				cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, false);
				CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
				read_time += timer.MicroSeconds();
				timer.Start();
				// Apply transformations (mirror, crop...) to the image
				int offset = batch->data_.offset(item_id);
				this->transformed_data_.set_cpu_data(prefetch_data + offset);
				this->data_transformer_->FaceImageTransform(cv_img, lines_[lines_id_].second.xy, &(this->transformed_data_));
				trans_time += timer.MicroSeconds();

				prefetch_label[item_id] = lines_[lines_id_].second.label;
				// go to the next iter
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					if (this->layer_param_.face_image_param().shuffle()) {
						ShuffleImages();
					}
				}
			}
			batch_timer.Stop();
			DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
			DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
			DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
		}


	}

	INSTANTIATE_CLASS(FaceImageLayer);
	REGISTER_LAYER_CLASS(FaceImage);

}  // namespace caffe
#endif  // USE_OPENCV
