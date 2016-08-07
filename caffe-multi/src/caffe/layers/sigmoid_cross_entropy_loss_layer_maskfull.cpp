#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossMaskFullLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossMaskFullLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  //////////////////////////////////// my change //////////////////////////////
  //LOG(INFO)<< "Till my code";
  mask_.ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void SigmoidCrossEntropyLossMaskFullLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();

 /////////////////////////////////////////
// const int num = bottom[0]->num();

  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
//  for (int i = 0; i < count; ++i) {
 //   loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
 //       log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
 // }
  /////////////////////////////// my change //////////////////////////////
 const Dtype* valid_mask = bottom[2]->cpu_data();
 Dtype* use_mask = mask_.mutable_cpu_data();
  

  if(bottom[1]->asum_data() > 0)
    negsig = 0;
  else
    negsig = 1;
  
  /** 
  this is version 1 
  if(negsig)
    for(int i=0; i<count; ++i) use_mask[i]=0;
  else{
      for(int i=0; i<count; ++i){
    if(target[i]==1){use_mask[i]=1;}
    else{
      if(valid_mask[i]==1){
          int testval = 0;
          testval = rand()%10000;
          if(testval<10)
              use_mask[i]=1;
          else
              use_mask[i]=0;
      }else{
        use_mask[i]=0;
      }
    }
  }
  }
  **/
 // int bp_neg_sample = 0;
 // int randval = 0;
 // int count1 = 0, count2 = 0;
  for(int i=0; i<count; ++i){use_mask[i]=0;}
  if(negsig==0){
  for(int i=0; i<count; ++i){
    if(valid_mask[i]==3){
        use_mask[i]=1;
     //   count1 ++;
      }
      if(valid_mask[i]==2){
        use_mask[i]=1;
     //   count2 ++;
      }  
    }
}else{
use_mask[0] = 1;
}

  /*
  int val1 = 0;
  int val0 = 0;
  int val2 = 0;
  int val3 = 0;
  for(int i=0; i<count; ++i)
  {
    if(valid_mask[i]==3)val3 = val3+1;
    if(valid_mask[i]==2)val2 = val2+1;
    if(valid_mask[i]==1)val1 = val1+1;
    if(valid_mask[i]==0)val0 = val0+1;
  }

 // LOG(INFO) << "bp neg around sample: " << bp_neg_sample;
  LOG(INFO) << "val0: " << val0;
  LOG(INFO) << "val1: " << val1;
  LOG(INFO) << "val2: " << val2;
  LOG(INFO) << "val3: " << val3;
  LOG(INFO) << "Suppose to be 56448. Total: " << val0+val1+val2+val3;
*/
  // IF ALL NEGATIVE, remove the influence

  for (int i = 0; i < count; ++i){
    if(use_mask[i]==1){
       loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }
  
  if(negsig)
    top[0]->mutable_cpu_data()[0] = 0;
  else
    top[0]->mutable_cpu_data()[0] = (loss/mask_.asum_data())*10;
 
  
///////////////////////////// my change /////////////////////////////////////
 //top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossMaskFullLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
   // const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    //////////////////////////////// my change /////////////////////////////////////
    const Dtype* use_mask = mask_.cpu_data();
    for (int i = 0; i < count; ++i){
      if(negsig){bottom_diff[i]=0;}
      if(use_mask[i]==0){bottom_diff[i]=0;}
    }
 //   LOG(INFO) << "Backward use_mask num" << mask_.asum_data();
  /////////////////////////////// my change /////////////////////////////////////
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / mask_.asum_data(), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossMaskFullLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossMaskFullLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLossMaskFull);

}  // namespace caffe
