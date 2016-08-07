#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
// This is changed from accuracy layer
template <typename Dtype>
void LabelMixLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Mix_Num_ = this->layer_param_.label_mix_param().mix_num();
}

template <typename Dtype>
void LabelMixLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  channel_num_ = bottom[0]->channels();
  inner_num_ = (bottom[0]->height()*bottom[0]->width());

  CHECK_EQ(channel_num_ , bottom[1]->channels())
      << "Number of labels channels must match number of mix; ";
 
  top[0]->Reshape(num_, (channel_num_*Mix_Num_), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void LabelMixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* mixtype = bottom[1]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();


  for(int n = 0; n < num_; ++n)
  {
    for(int i = 0; i< channel_num_; ++i)
    {
      mixtype[n*channel_num_+i] = mixtype[n*channel_num_+i]+13*i;
    }
  }


  int output_channel = top[0]->channels();
 // LOG(INFO) << "output_channel: " <<output_channel;
  int output_num = top[0]->count();
  for(int i=0; i<output_num; ++i){top_data[i]=0;}

  int bottom_idx = 0, top_idx = 0;
  for (int n = 0; n < num_; ++n) 
  {
    for (int i = 0; i < channel_num_; ++i) 
    {
      int this_mix = mixtype[n*channel_num_+i];
    //  LOG(INFO) << "n: " << n << " this_mix" << this_mix;
      for(int j = 0; j < inner_num_ ; ++j)
      {
        bottom_idx = n*channel_num_*inner_num_ + i*inner_num_ + j;
        top_idx    = n*output_channel*inner_num_ + (this_mix-1)*inner_num_ + j;
        top_data[top_idx] = bottom_data[bottom_idx];
      }
    }
  }
}

INSTANTIATE_CLASS(LabelMixLayer);
REGISTER_LAYER_CLASS(LabelMix);

}  // namespace caffe
