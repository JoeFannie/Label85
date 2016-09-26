#ifndef CAFFE_TXTFEATDATA_LAYER_HPP_
#define CAFFE_TXTFEATDATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/face_image_layer.hpp"

namespace caffe {

  template <typename Dtype>
  class TxtFeatDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
	  explicit  TxtFeatDataLayer(const LayerParameter& param)
		  : BasePrefetchingDataLayer<Dtype>(param) {}
	  virtual ~TxtFeatDataLayer();
	  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  virtual inline const char* type() const { return "TxtFeatData"; }
	  virtual inline int ExactNumBottomBlobs() const { return 0; }
	  virtual inline int ExactNumTopBlobs() const { return 2; }
  protected:
	  shared_ptr<Caffe::RNG> prefetch_rng_;
	  virtual void ShuffleTxtFeats();
	  virtual void load_batch(Batch<Dtype>* batch);
	  virtual bool ReadBinTxtFeat(const string& filename,
		  const int feat_dim, float *buffer);
	  virtual bool ReadTxtFeat(const string& filename,
		  const int feat_dim, float *buffer);
	  //virtual void get_one_triplet(int curr_line_id, int *res);
	  //virtual int  find_np_sample(int anchor_idx, int anchor_label, bool find_pos);
	  virtual int DisturbLabel(const int label, const float alpha, const int max_label);
	  vector<std::pair<std::string, FACE_IMAGE_INFO> > lines_;
	  int lines_id_;
	  int feat_dim_;
	  int b_bin_file_;
	  float dislab_alpha_;
	  int max_label_;
	  shared_ptr<Blob<float> > data_buf_;
  };


}  // namespace caffe

#endif  // CAFFE_TXTFEATDATA_LAYER_HPP_
