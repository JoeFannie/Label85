#ifndef CAFFE_FACE_IMAGE_LAYER_HPP_
#define CAFFE_FACE_IMAGE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  /**
    * @brief Provides data to the Net from image files.
    *
    * TODO(dox): thorough documentation for Forward and proto params.
    **/
  typedef struct _FACE_IMAGE_INFO_
  {
	  int xy[4];
	  int label;
  }FACE_IMAGE_INFO;
  template <typename Dtype>
  class FaceImageLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
	  explicit  FaceImageLayer(const LayerParameter& param)
		  : BasePrefetchingDataLayer<Dtype>(param) {}
	  virtual ~FaceImageLayer();
	  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  virtual inline const char* type() const { return "FaceImage"; }
	  virtual inline int ExactNumBottomBlobs() const { return 0; }
	  virtual inline int ExactNumTopBlobs() const { return 2; }
  protected:
	  shared_ptr<Caffe::RNG> prefetch_rng_;
	  virtual void ShuffleImages();
	  virtual void load_batch(Batch<Dtype>* batch);

	  virtual void get_one_triplet(int curr_line_id, int *res);
	  virtual int  find_np_sample(int anchor_idx, int anchor_label, bool find_pos);
          
	  vector<std::pair<std::string, FACE_IMAGE_INFO> > lines_;
	  int lines_id_;
  };
}  // namespace caffe

#endif  // CAFFE_FACE_IMAGE_LAYER_HPP_
