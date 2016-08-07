#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  const SoftmaxWithLossMaskParameter& mask_param = this->layer_param_.softmax_mask_param();
  use_neg_ = mask_param.use_neg();
  use_inria_ = mask_param.use_inria();
  inria_neg_ratio_ = mask_param.inria_neg_ratio();
  neg_ratio_ = mask_param.neg_ratio();
  neg_weight_ = mask_param.neg_weight();

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLossMaskLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = 
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  channel_num_ = bottom[0]->count(softmax_axis_, softmax_axis_+1); //////////////my change /////////////
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_ * (channel_num_-1), bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  label_.ReshapeLike(*bottom[0]);   // 19 channel
  use_mask_.ReshapeLike(*bottom[2]);
  bp_weight_.ReshapeLike(*bottom[2]);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}





//------------------------------Forward_cpu -----------------------------------------
template <typename Dtype>
void SoftmaxWithLossMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
// The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data(); /// prob stores the output probability predictions from the SoftmaxLayer.  
  //----------------- preprocess label -----------------------
  const Dtype* rawlabel = bottom[1]->cpu_data();
  float thresh_ = 2.5;
  
  //---------- marginalized map  -----------------------
  int margin_map[outer_num_][inner_num_]; 
  for(int i=0;i<outer_num_;++i) {for(int j = 0; j<inner_num_ ; j++) margin_map[i][j]=0; }

  int labelsum[outer_num_]; 
  for(int i=0;i<outer_num_;i++){labelsum[i] = 0;}

  for (int i = 0; i < outer_num_; ++i) { 
    for(int d = 0; d < channel_num_-1; ++d) {
       for (int j = 0; j < inner_num_; ++j) { 
        margin_map[i][j] += static_cast<int>(rawlabel[i * inner_num_* (channel_num_-1) + d*inner_num_ + j]);
        labelsum[i] += static_cast<int>(rawlabel[i * inner_num_* (channel_num_-1) + d*inner_num_ + j]);
      }
    }
  }

  /*//debug margin_map
  for (int n=0; n<outer_num_;++n)
  {
    for (int j = 0; j < inner_num_; ++j)
    {
      if (margin_map[n][j]>0)
        LOG(INFO) << "N: " << n << " j: " << j << "positve_value" << margin_map[n][j];
    }
    LOG(INFO) << "LABELSUM " << labelsum[n];
  } //debug */

  // --------------- give negatitve value ------------------------------------


 // LOG(INFO) << "channel_num_ " << channel_num_;

  int positive_channel_num = channel_num_-1; 
  Dtype* label = label_.mutable_cpu_data(); 
  for(int i=0;i<label_.count();++i)label[i]=0;

  int label_indx = 0;
  float margin_val = 0,raw_label_value=0;
  
  for (int i = 0; i < outer_num_; ++i) 
  {
    for (int j = 0; j < inner_num_; ++j) 
    {
      if (margin_map[i][j]==0){  // this is negative sample
        label[i * inner_num_ * channel_num_ + (channel_num_-1) * inner_num_ +j]=1;  
      //  debug_neg_ct++;
      }
      else{
          for(int d = 0; d < positive_channel_num ; ++d)
          {
              raw_label_value = rawlabel[i * inner_num_* positive_channel_num + d*inner_num_ + j];
              if(raw_label_value>0)
              {
                label_indx = i * inner_num_* channel_num_ + d*inner_num_ + j;
                label[label_indx] = raw_label_value;
                margin_val = static_cast<float>(margin_map[i][j]);
                label[label_indx] = label[label_indx]/margin_val;  //normalize by number of samples
              }
             // debug_pos_ct ++;
          }
      }
      DCHECK_GE(label[i * inner_num_ * channel_num_ + positive_channel_num * inner_num_ +j], 0);
      DCHECK_LE(label[i * inner_num_ * channel_num_ + positive_channel_num * inner_num_ +j], 1);
    }
  }
  CHECK_EQ(label_.count(), prob_.count()) 
    << "Number of predictions != Number of label";
/*
    for (int n = 0; n < outer_num_; ++n) {
      for(int i=0; i< channel_num_; ++ i){
         for (int j = 0; j < inner_num_; ++j) {
        if(label[n*channel_num_*inner_num_+i*inner_num_+j]>0) LOG(INFO) << "sample: " << n << " channel: " << i << " num: " << j;
        }
      }
    }
  // debug unit 

  // debug unit stop here */
 

  // ------------ caculate the used mask and bp value -----------------------
  Dtype* mask = use_mask_.mutable_cpu_data();
  Dtype* bp_weight = bp_weight_.mutable_cpu_data();
  int labelsize = bottom[2]->count();
  for(int i = 0; i<labelsize; i++){mask[i]=0;} //mask init as 0
  for(int i = 0; i<labelsize; i++){bp_weight[i]=1;} //bp_weight init as 1

  int randval = 0, negsig = 0, neg_thresh = 0, inria_neg_thresh = 0;

  if(use_neg_) {neg_thresh = 10000*neg_ratio_;};
  if(use_inria_) {inria_neg_thresh = 10000*inria_neg_ratio_;};
  const Dtype* valid_mask = bottom[2]->cpu_data();

  //LOG(INFO) << "use_inria_ " << use_inria_;

  for(int i = 0; i<outer_num_; i++)
  {
    if(labelsum[i]==0){
     // LOG(INFO) << "find a inria negative sample at " << i;
      negsig = 1;
    }
    else{
      negsig = 0;
    }
    
    if(negsig)
    {  /// this is a negative sample
      for(int j = 0; j<inner_num_; j++){
        randval = rand()%10000;
        if(randval<inria_neg_thresh){
          mask[i*inner_num_+j] = 2.9;
          bp_weight[i*inner_num_+j] = neg_weight_;
        }
      }
    }
    else
    { /// this is a positive sample
      for(int j = 0; j<inner_num_; j++)
      {
        if(valid_mask[i*inner_num_+j]==3)
        {
          mask[i*inner_num_+j] = 3;
        }
        if(valid_mask[i*inner_num_+j]==2)
        {
          randval = rand()%10000;
          if(randval<neg_thresh)
          {
            bp_weight[i*inner_num_+j] = neg_weight_;
            mask[i*inner_num_+j] = 2.9;
          }
        }
      }
    }
  }

 // LOG(INFO) << "batch label value p/n: " << debug_pos_ct << " " << debug_neg_ct;

  // --------------------- caculate the loss -------------------------------------
  int count2=0;
  Dtype loss = 0,prob_val = 0;
 // LOG(INFO) << "outer_num_ " << outer_num_;
  for (int i = 0; i < outer_num_; ++i) 
  {
    for(int d = 0; d < channel_num_; ++d) 
    {
       for (int j = 0; j < inner_num_; ++j) 
       {

        int mask_idx = i*inner_num_+j;
        if(mask[mask_idx]>=thresh_){
          
          int indx = i * inner_num_ * channel_num_ + d * inner_num_ + j;
          const float label_value = static_cast<float>(label[indx]);
          DCHECK_GE(label_value, 0);  // label_val >=0
          DCHECK_LT(label_value, prob_.shape(softmax_axis_)); // label_val <= prob_.shape(softmax_axis_)
          prob_val = prob_data[indx];
          
          if(label_value>0){
            //LOG(INFO) << "lOCATE " << indx << " " << label_value;
            loss -= label_value*(log(static_cast<float>(std::max(prob_val,Dtype(FLT_MIN))))-log(label_value));
           // count2++;
          }
          //count++;
        }
       }
     }
   }
    for (int i = 0; i < outer_num_; ++i){
      for (int j = 0; j < inner_num_; ++j) 
      {
        int mask_idx = i*inner_num_+j;
        if(mask[mask_idx]>=thresh_)
          count2++;
      }
    }
 // LOG(INFO) << "num of BP samples: " << count/19 ;
 //LOG(INFO) << "num of calculated samples: " << count2;
  // -------------------- normalize factors ---------------------
  if(count2==0){count2 = 1;}
  //LOG(INFO) << "count in forward: " << count;
//  LOG(INFO) << "count in forward: " << count;

  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count2;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}






template <typename Dtype>
void SoftmaxWithLossMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    //------------- my changes --------------------
    const Dtype* label = label_.cpu_data();
    const Dtype* mask = use_mask_.cpu_data();
    const Dtype* bp_weight = bp_weight_.cpu_data();
    //-----------------------------------------------
    caffe_copy(prob_.count(), prob_data, bottom_diff);

    int indx=0,count=0;
    float thresh_ = 2.5;
  
  // int bp_weight_negative_count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for(int d = 0; d < channel_num_; ++d) {
        for (int j = 0; j < inner_num_; ++j) {
          indx = i * inner_num_ * channel_num_ + d * inner_num_ + j;

          int mask_idx = i*inner_num_+j;
          if(mask[mask_idx]>=thresh_){
           
            const float label_value = static_cast<float>(label[indx]); 
            bottom_diff[indx] -= label_value;
            if(bp_weight[mask_idx]!=1){// nomalize negative sample
              bottom_diff[indx] = bottom_diff[indx]*bp_weight[mask_idx];
             //bp_weight_negative_count++;
            }
            count++;
          }else{
            bottom_diff[indx] = 0;
         //   debug_count++;
          }
        }
      }
    }

    if(count==0){
      count = 1;
    }
    else
      count = count/channel_num_;
    
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}
//LOG(INFO) << "CPU_ONLY: " << CPU_ONLY;
#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossMask);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossMaskLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossMask);

}  // namespace caffe
