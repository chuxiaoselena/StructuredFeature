# StructuredFeature
This is the code for our work "Structured Feature Learning for Pose estimation"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
If you use it, please cite our work
@inproceedings{chu2016structure,
  title={Structured Feature Learning for Pose Estimation},
  author={Chu, Xiao and Ouyang, Wanli and Li,Hongsheng and Wang, Xiaogang},
  booktitle={CVPR},
  year={2016}
}

Our project is written based on Xianjie Chen's code.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Steps to get your own model:
	1. Make caffe (We write our own layer for loss, channel dropout and mix interpolation, if 		you are not going to use these functions, you can use your own caffe)
	2. run "Data_prepare.m" to generate LMDB requires
	3. run "Baseline.sh" to train the model
	3. select the best model for test
	4. run "TestModel.m" to see the results.

Model release:
	We provide a model we trained on LSP dataset. (itr = 3250)
	If you are going to test this model, set the variable "test_our_provided_model" to true.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%R%%%%%%%%%%%%%%%%%
version: v2
 
In v1, the presave files: "cache/lsp/lsp_fconv/lsp_datainfo/Cropped_train.mat" and "cache/lsp/lsp_fconv/lsp_datainfo/Cropped_val.mat" prevent program from generating your own train and validiation data. In this version, we recetified this problem.
If you have alread down loaded v1, you can remove the two files mentioned to train your own model.
	
