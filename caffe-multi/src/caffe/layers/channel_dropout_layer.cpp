// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ChannelDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void ChannelDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(),bottom[0]->channels(),1,1);
}

template <typename Dtype>
void ChannelDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int inner_num_ = (bottom[0]->height())*(bottom[0]->width());
  CHECK_EQ(num * inner_num_ * channels, bottom[0]->count());

  //LOG(INFO) << "Mask Number" <<  rand_vec_.count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    int idx=0;
    int mask_val=0;
    caffe_rng_bernoulli(channels*num, 1. - threshold_, mask);
    for(int i = 0; i< num; ++i)
    {
      for(int j=0; j< channels; ++j)
      {
        for(int k = 0; k<inner_num_; ++k)
        {
          idx = i*channels*inner_num_+j*inner_num_+k;
          mask_val = mask[i*channels+j];
          top_data[idx] = bottom_data[idx] * mask_val * scale_;
         // LOG(INFO) <<"Forward: locate: " << i << " " << j << " " << k << " "
        //  << "Mask val " << mask_val << "  top_data: " << top_data[idx];
        }
      }
    }
    //for (int i = 0; i < count; ++i) {
    //  top_data[i] = bottom_data[i] * mask[i] * scale_;
    //}
  } 
  else { // do not use dropout at test stage
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void ChannelDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {

      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int inner_num_ = (bottom[0]->height())*(bottom[0]->width());

      const unsigned int* mask = rand_vec_.cpu_data();
      int idx=0, mask_val =0;
      for(int i = 0; i< num; ++i)
      {
        for(int j=0; j< channels; ++j)
        {
          for(int k = 0; k<inner_num_; ++k)
          {
            idx = i*channels*inner_num_+j*inner_num_+k;
            mask_val = mask[i*channels+j];
            bottom_diff[idx] = top_diff[idx] * mask_val * scale_;
       //     LOG(INFO) <<"Backward: locate: " << i << " " << j << " " << k << " "
       //   << "Mask val " << mask_val << "  top_data: " << bottom_diff[idx];
          }
        }
      }

      //for (int i = 0; i < count; ++i) {
       // bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      //}
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ChannelDropoutLayer);
#endif

INSTANTIATE_CLASS(ChannelDropoutLayer);
REGISTER_LAYER_CLASS(ChannelDropout);

}  // namespace caffe
