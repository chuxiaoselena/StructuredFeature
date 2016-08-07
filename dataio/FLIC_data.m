function [pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = FLIC_data()
conf = global_conf();
pa = conf.pa;
p_no = numel(pa);
cachedir = conf.cachedir;
assert(isequal(conf.dataset, 'flic'));
cls = [conf.dataset, num2str(p_no), '_data'];
rng(5);

try
  load([cachedir cls]);
%   load nofound.mat
catch
  % annotations we used
  joint_order = [17,4,5,6,10,1,2,3,7];
  % always transfer to 10 joints.
  I = [1    2   2   3  4  5  6  7  8  9   10];
  J = [1    2   6   2  3  4  5  6  7  8   9];
  A = [1   1/2 1/2  1  1  1  1  1  1  1   1];
  Trans_all = full(sparse(I,J,A,10,9));
  % -------------------
  % grab positive annotation and image information
  flic_imgs = './dataset/FLIC/images/%s';
  if ~exist('./dataset/FLIC', 'dir')
    error('Please downlad FLIC dataset from http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC');
  end
  flic_anno = parload('./dataset/FLIC/examples.mat', 'examples'); 
  % use official observer-centric annotations
  flic_anno = flip_backwards_facing_groundtruth(flic_anno); 
  
  trainval_frs_pos = ([flic_anno.istrain]); % training frames for positive
  test_frs_pos = ([flic_anno.istest]); % testing frames for positive
  assert(~any(trainval_frs_pos & test_frs_pos));
  assert(all(trainval_frs_pos | test_frs_pos));
  
  trainval_frs_neg = 615:1832;  % training frames for negative
  
  % ------------------ original images -------------------
  num = numel(flic_anno);
  all_pos = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
    'r_degree', cell(num, 1), 'isflip', cell(num,1), 'torsobox', cell(num,1));
  for ii = 1:numel(flic_anno)
    fr = flic_anno(ii);
    % ------------------
    all_pos(ii).im = sprintf(flic_imgs, fr.filepath);
    all_pos(ii).joints = Trans_all * fr.coords(1:2,joint_order)';
    all_pos(ii).r_degree = 0;
    all_pos(ii).isflip = 0;
    all_pos(ii).torsobox = fr.torsobox;
  end
  % -------------------
  % create ground truth joints for model training
  % We augment the original 10 joint positions with midpoints of joints,
  % defining a total of 18 joints
  switch p_no
    case 10
      Trans = eye(10,10);
      mirror = [1, 2, 7, 8, 9, 10, 3, 4, 5, 6];
    case 18
      I = [1    2   3   4   4   5   6   6   7   8   8   9   9   10 ...
        11   12  12   13  14 14  15  16  16  17   17   18];
      J = [1    2   3   3   4   4   4   5   5   3   6   3   6   6  ...
        7     7   8   8   8   9   9   7  10   7   10   10];
      A = [1    1   1  1/2 1/2  1  1/2  1/2 1  2/3 1/3 1/3 2/3  1 ...
        1    1/2 1/2  1  1/2 1/2  1  2/3 1/3 1/3 2/3   1];
      Trans = full(sparse(I,J,A,18,10));
      mirror = [1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10];
    otherwise
      error('p_no = %d is not supported!!', p_no);
  end
  pos_trainval = all_pos(trainval_frs_pos);
  pos_test = all_pos(test_frs_pos);
  % -------- generate constr_bbx for test --------
  for ii = 1:numel(pos_test)
    torsobox = pos_test(ii).torsobox;
    
    height = torsobox(4) - torsobox(2) + 1;
    width = torsobox(3) - torsobox(1) + 1;
    % --- Note: there are actually some error in the FLIC annotations ----
    constr_bbx = torsobox + [-width*2/3, -height*1/2, +width*2/3, +height/4];
    neck_xy = pos_test(ii).joints(2,:);
    assert(neck_xy(1) > constr_bbx(1) && neck_xy(1) < constr_bbx(3) ...
      && neck_xy(2) > constr_bbx(2) && neck_xy(2) < constr_bbx(4));
    pos_test(ii).constr_bbx = constr_bbx;
    full_bbx = flic_tbox2ubbox(torsobox);
    pos_test(ii).full_bbx = full_bbx;
  end
  % --------- handle trainval ----------------
  for ii = 1:numel(pos_trainval)
    pos_trainval(ii).joints = Trans * pos_trainval(ii).joints;
  end
  % --------- flip trainval images --------
  pos_trainval = add_flip(pos_trainval, mirror);
  % ------- init dataset specific parameters --------
  [pos_trainval, tsize] = init_scale(pos_trainval, pa, conf.step);
  % --------- rotate trainval images ---------
  degree = conf.degree;
  assert(numel(unique(degree)) == numel(degree));
  pos_trainval = add_rotate(pos_trainval, degree);
  % -------- split train, val -----------------
  val_id = randperm(numel(pos_trainval), 2000);
  train_id = true(numel(pos_trainval), 1); train_id(val_id) = false;
  pos_train = pos_trainval(train_id); pos_val = pos_trainval(val_id);
  
  % ------ neagtive images -----------
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
  save([cachedir cls],'pos_train','pos_val','pos_test','neg_train','neg_val','tsize');
end

