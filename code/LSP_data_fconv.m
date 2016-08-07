function [pos_train, pos_val,neg_train, neg_val] = LSP_data_fconv(cachedir)
conf = global_conf();
pa = conf.pa;
p_no = numel(pa);
% cachedir = conf.cachedir;
assert(isequal(conf.dataset, 'lsp'));

cls = [conf.dataset, num2str(p_no), '_data_fconv'];
rng(5);

try
    load([cachedir cls]);
catch
    trainval_frs_pos = 1:1000;      % training frames for positive
    test_frs_pos = 1001:2000;   % testing frames for positive
    
    trainval_frs_neg = 615:1832;    % training frames for negative
    joint_order = [14,13,9,8,7,3,2,1,10,11,12,4,5,6];
    % -------------------
    % grab positive annotation and image information
    lsp_imgs = './dataset/LSP/images/im%04d.jpg';
    if ~exist('./dataset/LSP', 'dir')
        error('Please downlad LSP dataset');
    end
    lsp_joints = parload('./dataset/LSP/joints.mat', 'joints');  % observer-centric annotation
    % convert to person-centric
    lsp_joints = lsp_pc2oc(lsp_joints);
    % ---------- original images --------
    frs_pos = cat(2, trainval_frs_pos, test_frs_pos);
    num = numel(frs_pos);
    all_pos = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
        'r_degree', cell(num, 1), 'isflip', cell(num,1));
    for ii = 1:numel(frs_pos)
        fr = frs_pos(ii);
        all_pos(ii).im = sprintf(lsp_imgs,fr);
        all_pos(ii).joints = lsp_joints(1:2,joint_order,fr)';
        all_pos(ii).r_degree = 0;
        all_pos(ii).isflip = 0;
    end
    % -------------------
    % create ground truth joints for model training
    % We augment the original 14 joint positions with midpoints of joints,
    % defining a total of 26 joints
    switch p_no
        case 14
            Trans = eye(14,14);
            mirror = [1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8];
        case 26
            % -------------------
            % create ground truth joints for model training
            % We augment the original 14 joint positions with midpoints of joints,
            % defining a total of 26 joints
            I = [1  2  3  4   4   5  6   6   7  8   8   9   9   10 11  11  12 13  13  14 ...
                15 16  16  17 18  18  19 20  20  21  21  22 23  23  24 25  25  26];
            J = [1  2  3  3   4   4  4   5   5  3   6   3   6   6  6   7   7  7   8   8 ...
                9  9  10  10  10  11  11 9   12  9   12  12 12  13  13 13  14  14];
            A = [1  1  1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1 ...
                1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1];
            Trans = full(sparse(I,J,A,26,14));
            mirror = [1,2,15,16,17,18,19,20,21,22,23,24,25,26,3,4,5,6,7,8,9,10,11,12,13,14];
        otherwise
            error('p_no = %d is not supported!!', p_no);
    end
    pos_trainval = all_pos(1 : numel(trainval_frs_pos));
    
    for ii = 1:numel(pos_trainval)
        pos_trainval(ii).joints = Trans * pos_trainval(ii).joints; % linear combination
    end
    
    % --------- flip trainval images --------
    pos_trainval = add_flip(pos_trainval, mirror);
    % ------- init dataset specific parameters --------
    pos_trainval = init_scale_lsp(pos_trainval, pa);
    % --------- rotate trainval images ---------
    degree = conf.degree;
    assert(numel(unique(degree)) == numel(degree));
    pos_trainval = add_rotate(pos_trainval, degree);
    % -------- split train, val -----------------
    val_id = randperm(numel(pos_trainval), 400);
    train_id = true(numel(pos_trainval), 1); train_id(val_id) = false;
    pos_train = pos_trainval(train_id); pos_val = pos_trainval(val_id);
    
    % -------------------
    % grab neagtive image information
    negims = './dataset/INRIA/%05d.jpg';
    if ~exist('./dataset/INRIA', 'dir')
        error('Please downlad INRIA dataset');
    end
    num = numel(trainval_frs_neg);
    neg = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
        'r_degree', cell(num, 1), 'isflip', cell(num,1));
    
    for ii = 1:num
        fr = trainval_frs_neg(ii);
        neg(ii).im = sprintf(negims,fr);
        neg(ii).joints = [];
        neg(ii).r_degree = 0;
        neg(ii).isflip = 0;
    end
    % -------- flip negatives ----------
    val_id = randperm(numel(neg), 500);
    train_id = true(numel(neg), 1); train_id(val_id) = false;
    neg_train = neg(train_id); neg_val = neg(val_id);
    
    neg_train = add_flip(neg_train, []);
    neg_val = add_flip(neg_val, []);
    save([cachedir cls],'pos_train','pos_val','neg_train','neg_val');
end


