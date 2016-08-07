function conf = lsp_conf(conf)
conf.dataset = 'lsp';
conf.step = 8;
conf.K = 13;
% conf.K = 11;
conf.NEG_N = 100;

conf.test_with_detection = false; % single response on one image

% for full body
% 26 part
conf.pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
d_step = 9;
conf.degree = [-180+d_step:d_step:-d_step,d_step:d_step:180];

% 14 part
% conf.pa = [0 1 2 3 4 3 6 7 2 9 10 9 12 13];
% d_step = 6;
% conf.degree = [-180+d_step:d_step:-d_step,d_step:d_step:180];

% ---------------- CNN files ------------------
conf.cnn.cnn_deploy_conv_file = './external/my_models/lsp/lsp_deploy_conv.prototxt';
conf.cnn.cnn_conv_model_file = './cache/lsp/fully_conv_net_by_net_surgery.caffemodel';
conf.cnn.cnn_deploy_file = './external/my_models/lsp/lsp_deploy.prototxt';
conf.cnn.cnn_model_file = './cache/lsp/lsp_iter_60000.caffemodel';

% ----- evaluation functions -----
% --- part sticks ---
% symmetry_part_id(i) = j, if part j is the symmetry part of i (e.g., the left
% upper arm is the symmetry part of the right upper arm).
conf.symmetry_part_id = [1,2,7,8,9,10,3,4,5,6];
% show the average pcp performance of each pair of symmetry parts.
conf.show_part_ids = find(conf.symmetry_part_id >= 1:numel(conf.symmetry_part_id));
conf.part_name = {'Head', 'Torso', 'U.arms', 'L.arms', 'U.legs', 'L.legs'};

% ---- joints ----
% the pair of reference joints is used to defined the scale of each pose.
conf.reference_joints_pair = [6, 9];     % right shoulder and left hip (from observer's perspective)
% symmetry_joint_id(i) = j, if joint j is the symmetry joint of i (e.g., the left
% shoulder is the symmetry joint of the right shoulder).
conf.symmetry_joint_id = [2,1,9,10,11,12,13,14,3,4,5,6,7,8];
conf.show_joint_ids = find(conf.symmetry_joint_id >= 1:numel(conf.symmetry_joint_id)); 
conf.joint_name = {'Head', 'Shou', 'Elbo', 'Wris', 'Hip', 'Knee', 'Ankle'};

conf.box2det = @lsp_box2det;
conf.joint2stick = @lsp_joint2stick;

% the DCNN inference method
% 1. impyra-> fully convolutional way, fast
% 2. impyra_exact-> exact inferent by croping windows, slightly higher performace, but slow for large images
conf.impyra_fun = @impyra_exact;
% conf.impyra_fun = @impyra;

