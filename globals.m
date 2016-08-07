function globals

if ~isdeployed
    addpath('./external/Dt_yy');
    addpath('./external/qpsolver');
    addpath('./dataio');
    addpath('./evaluation');
    addpath('./visualization');
    addpath('./tools');
    addpath('./external');
    addpath('./code');
    conf = global_conf();
    caffe_root = conf.caffe_root;
    if exist(fullfile(caffe_root, '/matlab'), 'dir')
        addpath(genpath(fullfile(caffe_root, '/matlab')));
    else
        warning('Please install Caffe in %s', caffe_root);
    end
    %
    %   if exist(fullfile(caffe_root, '/matlab/caffe'), 'dir')
    %     addpath(fullfile(caffe_root, '/matlab/caffe'));
    %   else
    %     warning('Please install Caffe in %s', caffe_root);
    %   end
end
