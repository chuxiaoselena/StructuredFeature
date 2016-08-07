function  net=matcaffe_init(use_gpu, model_def_file, model_file,gpu_id)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize the new version of matcaffe

if nargin < 4
  % By default use gpu 0
  gpu_id = 0;
end

if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
end

if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
end

if use_gpu == 0
    caffe.set_mode_cpu();
else
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
end

net = caffe.Net(model_def_file, model_file, 'test'); % create net and load weights

fprintf('Done with init\n');
