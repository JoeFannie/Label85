/*
* =====================================================================================
*
*       Filename:  triplet_loss_layer.cpp
*
*    Description:
*
*        Version:  1.0
*        Created:  2015Äê08ÔÂ07ÈÕ 16Ê±31·Ö56Ãë
*       Revision:  none
*       Compiler:  gcc
*
*         Author:  YuanYang (), bengouawu@gmail.com
*        Company:  SUNTEKPCI
*
* =====================================================================================
*/

/*
* triplet_loss_layer.cpp
*/

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
//#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"


namespace caffe {

  template <typename Dtype>
  void TripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	CHECK_EQ(bottom[0]->num() % 3, 0);
	CHECK_EQ(bottom[0]->height(), 1);
	CHECK_EQ(bottom[0]->width(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	CHECK_EQ(bottom[1]->channels(), 1);

	diff_ap_.Reshape(bottom[0]->num() / 3, bottom[0]->channels(), 1, 1);
	diff_an_.Reshape(bottom[0]->num() / 3, bottom[0]->channels(), 1, 1);
	diff_pn_.Reshape(bottom[0]->num() / 3, bottom[0]->channels(), 1, 1);

	diff_sq_ap_.Reshape(bottom[0]->num() / 3, bottom[0]->channels(), 1, 1);
	diff_sq_an_.Reshape(bottom[0]->num() / 3, bottom[0]->channels(), 1, 1);
	dist_sq_ap_.Reshape(bottom[0]->num() / 3, 1, 1, 1);
	dist_sq_an_.Reshape(bottom[0]->num() / 3, 1, 1, 1);


	top[0]->Reshape(1, 1, 1, 1);

	// vector of ones used to sum along channels
	summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
	for (int i = 0; i < bottom[0]->channels(); ++i)
		summer_vec_.mutable_cpu_data()[i] = Dtype(1);
	dist_binary_.Reshape(bottom[0]->num() / 3, 1, 1, 1);
	for (int i = 0; i < bottom[0]->num() / 3; ++i)
		dist_binary_.mutable_cpu_data()[i] = Dtype(1);
  }

  template <typename Dtype>
  void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	  int count = bottom[0]->count() / 3;

	  const Dtype * anchor_data = bottom[0]->cpu_data();
	  const Dtype * pos_data = bottom[0]->cpu_data() + count;
	  const Dtype * neg_data = bottom[0]->cpu_data() + 2 * count;

	  const Dtype *label = bottom[1]->cpu_data();
	  for (int i = 0; i < bottom[1]->num() / 3; i++)
	  {
		  CHECK_EQ(label[i], label[i + bottom[1]->num() / 3]);
		  CHECK_NE(label[i], label[i + 2 * bottom[1]->num() / 3]);
	  }


	  caffe_sub(
		  count,
		  anchor_data,  // a
		  pos_data,  // p
		  diff_ap_.mutable_cpu_data());  // a_i-p_i
	  caffe_sub(
		  count,
		  anchor_data,  // a
		  neg_data,  // n
		  diff_an_.mutable_cpu_data());  // a_i-n_i
	  caffe_sub(
		  count,
		  pos_data,  // p
		  neg_data,  // n
		  diff_pn_.mutable_cpu_data());  // p_i-n_i
	  const int channels = bottom[0]->channels();

	  Dtype margin = this->layer_param_.triplet_loss_param().margin();


	  for (int i = 0; i < bottom[0]->num() / 3; ++i) {
		  dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
			  diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));
		  dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
			  diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));
	  }


	  Dtype loss(0.0);
	  for (int i = 0; i < bottom[0]->num() / 3; ++i) {
		  dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
			  diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));
		  dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
			  diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));
		  Dtype mdist = std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));
		  loss += mdist;
		  if (mdist < Dtype(1e-6)) {
			  //prepare for backward pass
			  caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i*channels));
			  caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));
			  caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i*channels));
		  }
	  }

	  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
  }

  template <typename Dtype>
  void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    //const Dtype* sampleW = bottom[3]->cpu_data();
	  int num = bottom[0]->num() / 3;
	  int channels = bottom[0]->channels();


	  Dtype* anchor_out = bottom[0]->mutable_cpu_diff();
	  Dtype* pos_out = bottom[0]->mutable_cpu_diff() + bottom[0]->count() / 3;
	  Dtype* neg_out = bottom[0]->mutable_cpu_diff() + 2 * bottom[0]->count() / 3;

	  Dtype lamda = this->layer_param_.triplet_loss_param().lamda();

	  const Dtype alpha = lamda * top[0]->cpu_diff()[0] /
		  static_cast<Dtype>(bottom[0]->num());
	  //const Dtype alpha = top[0]->cpu_diff()[0];

	  for (int j = 0; j < num; ++j)
	  {
		  //anchor
		 caffe_cpu_axpby(channels,
				  alpha*Dtype(-1),
				  diff_pn_.cpu_data() + (j*channels),
				  Dtype(0.0),
				  anchor_out + (j*channels));

		//positive
		caffe_cpu_axpby(channels,
				  alpha*Dtype(-1),
				  diff_ap_.cpu_data() + (j*channels),
				  Dtype(0.0),
				  pos_out + (j*channels));

		//negitive
		caffe_cpu_axpby(channels,
				  alpha,
				  diff_an_.cpu_data() + (j*channels),
				  Dtype(0.0),
				  neg_out + (j*channels));
	  }

  }

#ifdef CPU_ONLY
  STUB_GPU(TripletLossLayer);
#endif

  INSTANTIATE_CLASS(TripletLossLayer);
  REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe

