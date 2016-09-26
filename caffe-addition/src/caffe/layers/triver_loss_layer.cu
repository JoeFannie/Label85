/*
* triver_loss_layer.cu
*
*/

#include <algorithm>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triver_loss_layer.hpp"


namespace caffe {

	template <typename Dtype>
	void TriVerLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		int count = bottom[0]->count() / 3;

		const Dtype * anchor_data = bottom[0]->gpu_data();
		const Dtype * pos_data = bottom[0]->gpu_data() + count;
		const Dtype * neg_data = bottom[0]->gpu_data() + 2 * count;

		const Dtype *label = bottom[1]->cpu_data();
		for (int i = 0; i < bottom[1]->num() / 3; i++)
		{
			CHECK_EQ(label[i], label[i + bottom[1]->num() / 3]);
			CHECK_NE(label[i], label[i + 2 * bottom[1]->num() / 3]);
		}

		//const Dtype* sampleW = bottom[3]->cpu_data();
		//const Dtype sampleW = Dtype(1);
		caffe_gpu_sub(
			count,
			anchor_data,  // a
			pos_data,  // p
			diff_ap_.mutable_gpu_data());  // a_i-p_i
		caffe_gpu_sub(
			count,
			anchor_data,  // a
			neg_data,  // n
			diff_an_.mutable_gpu_data());  // a_i-n_i
		caffe_gpu_sub(
			count,
			pos_data,  // p
			neg_data,  // n
			diff_pn_.mutable_gpu_data());  // p_i-n_i
		const int channels = bottom[0]->channels();

		caffe_gpu_powx(
			count,
			diff_ap_.mutable_gpu_data(),  // a_i-p_i
			Dtype(2),
			diff_sq_ap_.mutable_gpu_data());  // (a_i-p_i)^2
		caffe_gpu_gemv(
			CblasNoTrans,
			bottom[0]->num()/3,
			bottom[0]->channels(),
			Dtype(1.0),                                         //alpha
			diff_sq_ap_.gpu_data(),  // (a_i-p_i)^2                // A
			summer_vec_.gpu_data(),                             // x
			Dtype(0.0),                                         //belta
			dist_sq_ap_.mutable_gpu_data());  // \Sum (a_i-p_i)^2  //y

		caffe_gpu_powx(
			count,
			diff_an_.mutable_gpu_data(),  // a_i-n_i
			Dtype(2),
			diff_sq_an_.mutable_gpu_data());  // (a_i-n_i)^2
		caffe_gpu_gemv(
			CblasNoTrans,
			bottom[0]->num()/3,
			bottom[0]->channels(),
			Dtype(1.0),                                         //alpha
			diff_sq_an_.gpu_data(),  // (a_i-n_i)^2                // A
			summer_vec_.gpu_data(),                             // x
			Dtype(0.0),                                         //belta
			dist_sq_an_.mutable_gpu_data());  // \Sum (a_i-n_i)^2  //y


		//------------------------------------------------------------------------------
		//calculate m_0
		Dtype m_0 = 0;
		int  m_cnt = 0;
		for (int i = 0; i < bottom[0]->num() / 3; ++i)
		{
			Dtype thd = dist_sq_ap_.cpu_data()[i];
			int cnt = 0;
			for (int j = 0; j < bottom[0]->num() / 3; ++j)
			{
				if (dist_sq_ap_.cpu_data()[i] <= thd)
				{
					++cnt;
				}
				if (dist_sq_an_.cpu_data()[i] > thd)
				{
					++cnt;
				}
			}
			if (cnt > m_cnt)
			{
				m_cnt = cnt;
				m_0 = thd;
			}
		}

		for (int i = 0; i < bottom[0]->num() / 3; ++i)
		{
			Dtype thd = dist_sq_an_.cpu_data()[i];
			int cnt = 0;
			for (int j = 0; j < bottom[0]->num() / 3; ++j)
			{
				if (dist_sq_ap_.cpu_data()[i] <= thd)
				{
					++cnt;
				}
				if (dist_sq_an_.cpu_data()[i] > thd)
				{
					++cnt;
				}
			}
			if (cnt > m_cnt)
			{
				m_cnt = cnt;
				m_0 = thd;
			}
		}
		//-----------------------------------------------------------

		Dtype k = this->layer_param_.triver_loss_param().k();
		
		m_ = k*m_0;
		//m_ = 20;

		Dtype loss(0.0);
		for (int i = 0; i < bottom[0]->num() / 3; ++i) {
			Dtype mdist = std::max(m_ - dist_sq_an_.cpu_data()[i], Dtype(0.0));
			loss_an_.mutable_cpu_data()[i] = mdist;
			loss += dist_sq_ap_.cpu_data()[i] + mdist;
		}

		loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;

	}

	template <typename Dtype>
	void TriVerLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int num = bottom[0]->num() / 3;
		int channels = bottom[0]->channels();


		Dtype* anchor_out = bottom[0]->mutable_gpu_diff();
		Dtype* pos_out = bottom[0]->mutable_gpu_diff() + bottom[0]->count() / 3;
		Dtype* neg_out = bottom[0]->mutable_gpu_diff() + 2 * bottom[0]->count() / 3;

		Dtype lamda = this->layer_param_.triver_loss_param().lamda();

		const Dtype alpha = lamda * top[0]->cpu_diff()[0] /
			static_cast<Dtype>(bottom[0]->num());


		for (int j = 0; j < num; ++j)
		{
			if (loss_an_.cpu_data()[j] > Dtype(1e-9))
			{
				//anchor
				caffe_gpu_axpby(channels,
					alpha*Dtype(-1),
					diff_pn_.gpu_data() + (j*channels),
					Dtype(0.0),
					anchor_out + (j*channels));

				//positive
				caffe_gpu_axpby(channels,
					alpha*Dtype(-1),
					diff_ap_.gpu_data() + (j*channels),
					Dtype(0.0),
					pos_out + (j*channels));

				//negitive
				caffe_gpu_axpby(channels,
					alpha,
					diff_an_.gpu_data() + (j*channels),
					Dtype(0.0),
					neg_out + (j*channels));
			}
			else{
				//anchor
				caffe_gpu_axpby(channels,
					alpha,
					diff_ap_.gpu_data() + (j*channels),
					Dtype(0.0),
					anchor_out + (j*channels));

				//positive
				caffe_gpu_axpby(channels,
					alpha*Dtype(-1),
					diff_ap_.gpu_data() + (j*channels),
					Dtype(0.0),
					pos_out + (j*channels));

				//negitive
				caffe_gpu_set(channels,
					Dtype(0.0),
					neg_out + (j*channels));
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(TriVerLossLayer);

}  // namespace caffe
