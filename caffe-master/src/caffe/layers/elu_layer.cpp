#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();
		Dtype alpha = this->layer_param_.elu_param().alpha();
		CHECK_GT(alpha, 0);
		for (int i = 0; i < count; ++i) {
			top_data[i] = (bottom_data[i] > 0) ? bottom_data[i] : (alpha * (exp(bottom_data[i]) - 1));
		}
	}

	template <typename Dtype>
	void ELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			Dtype alpha = this->layer_param_.elu_param().alpha();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int count = bottom[0]->count();
			for (int i = 0; i < count; ++i) {
				//bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
				//	+ negative_slope * (bottom_data[i] <= 0));
				bottom_diff[i] = top_diff[i] * ((bottom_data[i]>0) ? 1 : (top_data[i] + alpha));
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(ELULayer);
#endif

	INSTANTIATE_CLASS(ELULayer);
	REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe
