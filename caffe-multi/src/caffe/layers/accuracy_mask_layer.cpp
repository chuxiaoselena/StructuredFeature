#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyMaskLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyMaskLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);


  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
      << "Number of masks must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of masks) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  //top[0]->Reshape(top_shape);
  mask_.ReshapeLike(*(bottom[0]));
}


template <typename Dtype>
void AccuracyMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  

  const Dtype* valid_mask = bottom[2]->cpu_data();
  Dtype* use_mask = mask_.mutable_cpu_data();

  int count0 = 0;
  count0 = dim*outer_num_;
  for(int i=0; i<count0; ++i){
    use_mask[i]=0;
  }


  
  for(int i=0; i<count0; ++i)
  {
      if(valid_mask[i]==3)
        use_mask[i]=1;
      else
        use_mask[i]=0;
  }
  
 // LOG(INFO) << "count0 " << count0;
//  LOG(INFO) << "use_mask_ " << mask_.asum_data();

  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      if(use_mask[i * inner_num_ + j]==1)
      {
                const int label_value =
                    static_cast<int>(bottom_label[i * inner_num_ + j]); // change to int
                if (has_ignore_label_ && label_value == ignore_label_) {
                  continue;
                }

                DCHECK_GE(label_value, 0);
                DCHECK_LT(label_value, num_labels);
                // Top-k accuracy
                std::vector<std::pair<Dtype, int> > bottom_data_vector;


                for (int k = 0; k < num_labels; ++k) {
                  bottom_data_vector.push_back(std::make_pair(
                      bottom_data[i * dim + k * inner_num_ + j], k));
                }
                std::partial_sort(
                    bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
                    bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
                // check if true label is in top k predictions

                for (int k = 0; k < top_k_; k++) {
                  if (bottom_data_vector[k].second == label_value) {
                    ++accuracy;
                    break;
                  }
                }

                ++count;
      }
    }
  }
  //LOG(INFO) << "outer_num_ " << outer_num_;
 // LOG(INFO) << "inner_num_ " << inner_num_;d
 // LOG(INFO) << "num_labels: " << num_labels;


  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyMaskLayer);
REGISTER_LAYER_CLASS(AccuracyMask);

}  // namespace caffe


