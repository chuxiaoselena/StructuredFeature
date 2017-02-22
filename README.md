Structured Feature Learning for Pose Estimation
============================

This is the code for our work [Structured Feature Learning for Pose estimation](https://arxiv.org/abs/1603.09065)


## Training
**Make caffe**: We write our own layer for loss, channel dropout and mix interpolation, if you are not going to use these functions, you can use your own caffe.
```bash
make matcaffe
```
**Get LMDB**: Run "Data_prepare.m" in matlab to generate LMDB requires
**Train the caffe model**: Run "Baseline.sh. You may need the pre-train fully convolutional [VGG-16](https://www.dropbox.com/s/he3qdxk6pspjct8/VGG_ILSVRC_16_layers_full_conv.caffemodel?dl=0) model.
```bash
./Baseline.sh
```
**Test**: Select the best model for testing, and run "TestModel.m" to see the results.

## Released models
We provide a [model](https://www.dropbox.com/s/2pvtcpnkg2yl8lx/lspmodel.caffemodel?dl=0) we trained on LSP dataset (itration = 3250). If you are going to test this model, please download it and put it in the location specified in code, and set the variable "test_our_provided_model" to true.

## Cite
If you use this code, please cite our work 
```bash
@inproceedings{chu2016structure, 
title={Structured Feature Learning for Pose Estimation}, 
author={Chu, Xiao and Ouyang, Wanli and Li,Hongsheng and Wang, Xiaogang}, 
booktitle={CVPR}, year={2016} 
}
```
Our project is written based on [Xianjie Chen's NIPS2014](http://www.stat.ucla.edu/~xianjie.chen/projects/pose_estimation/pose_estimation.htm)

