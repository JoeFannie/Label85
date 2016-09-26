#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_metric_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyMetricLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void AccuracyMetricLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  outer_num_ = bottom[0]->shape(0);
  dim_ = bottom[0]->shape(1);

  dist_sq_.Reshape(outer_num_, 1, 1, 1);
  dot_.Reshape(outer_num_, outer_num_, 1, 1);
  ones_.Reshape(outer_num_, 1, 1, 1);
  for (int i = 0; i < outer_num_; ++i)
  {
      ones_.mutable_cpu_data()[i] = Dtype(1);
  }

  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracyMetricLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  for (int i = 0; i < bottom[0]->num(); i++){
      dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(dim_, bottom[0]->cpu_data() + (i*dim_), bottom[0]->cpu_data() + (i*dim_));
  }

  Dtype dot_scaler(-2.0);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, outer_num_, outer_num_, dim_, dot_scaler, bottom_data, bottom_data, (Dtype)0., dot_.mutable_cpu_data());

  // add ||x_i||^2 to all elements in row i
  for (int i=0; i<outer_num_; i++){
      caffe_axpy(outer_num_, dist_sq_.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i*outer_num_);
  }

  // add the norm vector to row i
  for (int i=0; i<outer_num_; i++){
      caffe_axpy(outer_num_, Dtype(1.0), dist_sq_.cpu_data(), dot_.mutable_cpu_data() + i*outer_num_);
  }


  const Dtype* dot_data = dot_.cpu_data();

  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {

      const int label_value = static_cast<int>(bottom_label[i]);
      int has_same_class = 0;

      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < outer_num_; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            dot_data[i*outer_num_ + k], k));
        has_same_class += (bottom_label[k] == label_value);
      }

      if (has_same_class < 2) continue; 

      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_ + 1,
          bottom_data_vector.end(), std::less<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      CHECK((bottom_data_vector[0].first < 1e-2) || (static_cast<int>(bottom_label[bottom_data_vector[0].second]) == label_value));
      for (int k = 1; k < top_k_ + 1; k++) {
        if (static_cast<int>(bottom_label[bottom_data_vector[k].second]) == label_value) {
          ++accuracy;
          break;
        }
      }
      ++count;
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyMetricLayer);
REGISTER_LAYER_CLASS(AccuracyMetric);

}  // namespace caffe
