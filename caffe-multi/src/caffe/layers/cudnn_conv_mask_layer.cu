#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

__global__ void sync_conv_groups_() { }

//using std::cout;
//using std::endl;

template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    size_t workspace_limit_bytes = this->kernel_h_ *
                                   this->kernel_w_ *
                                   this->channels_ *
                                   sizeof(int) + 1;

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      cudnnConvolutionFwdAlgo_t algo;

      // pick the convolution algorithm
      // TODO(shelhamer) this should be done during reshape
      // TODO(shelhamer) the choice of automatic or manual algorithm picking
      // should be exposed in proto
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,  // memoryLimitInBytes,
        &algo));

      // get minimum size of the workspace needed for the desired algorithm
      size_t workspaceSizeInBytes_temp = 0;

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        algo,
        &workspaceSizeInBytes_temp));

      if (workspaceSizeInBytes_temp > workspaceSizeInBytes) {
        workspaceSizeInBytes = workspaceSizeInBytes_temp;
        // free the existing workspace and allocate a new (larger) one
        cudaFree(this->workspace);
        cudaError_t err = cudaMalloc(&(this->workspace), workspaceSizeInBytes);
        if (err != cudaSuccess) {
          // force zero memory path
          algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          workspace = NULL;
          workspaceSizeInBytes = 0;
        }
      }

      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + weight_offset_ * g,
            conv_descs_[i],
            algo, workspace, workspaceSizeInBytes,
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)

    //calculate the mask 
    if(first_time_)
    {
      //LOG(INFO) << "First time forward"; 
      const Dtype* model_w_ = NULL;
    model_w_ = this->blobs_[0]->gpu_data();
    int dim1 = this->blobs_[0]->count(0, 1);
    int dim2 = this->blobs_[0]->count(1, 2);
    int kernel_num = this->blobs_[0]->count(2,4);
    Dtype* mask = mask_.mutable_cpu_data();
    for(int n_out = 0; n_out<dim1; n_out++)
    {
      for(int n_in = 0; n_in<dim2; n_in++)
      {
      Dtype flag = 0;
      caffe_gpu_asum(kernel_num,model_w_ + n_out*dim2*kernel_num + n_in*kernel_num, &flag);
      if(flag<0.0000000001) {
        //  LOG(INFO) << "zero: n_out:" << n_out << " n_in:" << n_in;
          caffe_set(kernel_num, (Dtype)(0.), mask + n_out*dim2*kernel_num + n_in*kernel_num);
      }
      else
      {
        caffe_set(kernel_num, (Dtype)(1.), mask + n_out*dim2*kernel_num + n_in*kernel_num);
      }
    }
    }
    first_time_ = false;  
    }
  



   sync_conv_groups_<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + weight_offset_ * g));
      }
      
      // My mask generator
      //Dtype* mask = mask_.gpu_data();
      caffe_gpu_mul(this->blobs_[0]->count(0, 4), weight_diff, mask_.gpu_data(),weight_diff);
     // int dim3 = this->blobs_[0]->count(2, 3);
     // int dim4 = this->blobs_[0]->count(3, 4);
     // int kernel_num = dim3*dim4;
      //Dtype zero_val = 0;
  //    const Dtype* weight_cpu = this->blobs_[0]->gpu_data();
    //  for(int n_out = 0; n_out<dim1; n_out++)
     // {
      //  for(int n_in = 0; n_in<dim2; n_in++)
      //  {
      //    Dtype flag = 0;
      //    caffe_gpu_asum(kernel_num,weight + n_out*dim2*kernel_num + n_in*kernel_num, &flag);
      //   if(flag<0.0000000001) {
      //      caffe_gpu_set(kernel_num, (Dtype)(0.), weight_diff + n_out*dim2*kernel_num + n_in*kernel_num);
      //    }
      //  }
     // }
      //cout << "end" << endl;


      // cout << "weight dim1: " << dim1 << endl;
      // cout << "weight dim2: " << dim2 << endl;
      // cout << "weight dim3: " << dim3 << endl;
      // cout << "weight dim4: " << dim4 << endl;

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
   sync_conv_groups_<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionMaskLayer);

}  // namespace caffe
#endif
