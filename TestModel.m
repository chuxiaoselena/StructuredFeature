globals;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();
cachedir = 'cache/lsp/';
if ~isdir(cachedir),mkdir(cachedir);end;
pa = conf.pa;
p_no = length(pa);
MIX = 13;
pos_test = LSP_data_test(cachedir);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------- load model + post processing ------------------------
name = 'lsp_basic';
deploy_file = './protofiles/deploy.prototxt';
model_file = './cache/lspmodel.caffemodel';% our provided model
% model_file = './cache/Strict_iter_3000.caffemodel';
UseGpu = 1;
gpu_id = 0;
net = matcaffe_init(UseGpu,deploy_file,model_file,gpu_id);

boxes = test_mix_postprocess_lsp(name,net,pos_test(1:5));

boxes = test_mix_postprocess_lsp(name,net,pos_test);
caffe.reset_all();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------------- evaluation -----------------------------------
for i = 1:length(boxes)
    if size(boxes{i},1)>1
        [~,maxid] = max(boxes{i}(:,end));
        boxes{i} = boxes{i}(maxid,:);
    end
end
eval_method = {'strict_pcp'};
% eval_method = {'strict_pcp', 'pdj'};
fprintf('=======%s=========\n',name);
fprintf('============= On test =============\n');
ests = conf.box2det(boxes, p_no);
% generate part stick from joints locations
for ii = 1:numel(ests)
    ests(ii).sticks = conf.joint2stick(ests(ii).joints);
    pos_test(ii).sticks = conf.joint2stick(pos_test(ii).joints);
end
show_eval(pos_test, ests, conf, eval_method);

% show_eval(pred, ests, conf, eval_method);
