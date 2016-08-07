function [box, scale, bdims, ctr] = flic_tbox2ubbox(tbox)
% Code from MODEC paper:
% @inproceedings{modec13,
%   title={Multimodal Decomposable Models for Human Pose Estimation},
%   author={Sapp, Benjamin and Taskar, Ben},
%   booktitle={In Proc. CVPR},
%   year={2013},
% }

ctr = [mean(tbox([1 3])); tbox(2)];
width = abs(diff(tbox([1 3])));
width = width*1.4;
scale = 50/width;
bdims = [-100 -50 100 100];
imgdims = [bdims(4)-bdims(2),bdims(3)-bdims(1)];
box = ctr([1 2 1 2])' + bdims/scale;
    
