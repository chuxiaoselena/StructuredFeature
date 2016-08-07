function cpatch = crop_patch(imdata, label, psize)
persistent NEG_N;

if isempty(NEG_N)
  conf = global_conf();
  NEG_N = conf.NEG_N;
end

RNG_MAX = 1e9 - 1;
is_negative = isempty(imdata.joints);

if ~is_negative
  % negative images
  joints = imdata.joints;
  scale_x = imdata.scale_x;
  scale_y = imdata.scale_y;
  glabel = label.global_id;
  
  if isfield(label, 'invalid')
    invalid = label.invalid;
  else
    invalid = false(size(joints,1), 1);
  end
  valid_ps = find(~invalid);
  p_no = numel(valid_ps);
  
  cpatch = struct('patch', cell(p_no, 1), 'label', cell(p_no, 1), 'rngkey', cell(p_no, 1));
  
  im = imreadx(imdata);
  
  for ii = 1:p_no
    pid = valid_ps(ii);
    bbx = [joints(pid,1)-scale_x, joints(pid,2)-scale_y, ...
      joints(pid,1)+scale_x, joints(pid,2)+scale_y];
    bbx = round(bbx);
    crop_im = subarray(im, bbx(2),bbx(4),bbx(1),bbx(3),1);
    crop_im = imresize(crop_im, psize);
    
    cpatch(ii).patch = crop_im;
    
    cpatch(ii).label = int32(glabel(pid));
    cpatch(ii).rngkey = randi(RNG_MAX,1,'int32');
  end
else
  % ------ sample N negatives from each negative images ----
  % negative images
  im = imreadx(imdata);
  neg_bbx = sample_negative(im, psize, NEG_N);
  b_n = size(neg_bbx, 1);
  cpatch = struct('patch', cell(b_n, 1), 'label', cell(b_n,1), 'rngkey', cell(b_n,1));
  
  for ii = 1:b_n
    crop_im = subarray(im, ...
      neg_bbx(ii,2),neg_bbx(ii,4),neg_bbx(ii,1),neg_bbx(ii,3),1);
    assert(size(crop_im,1) == psize(1) && size(crop_im,2) == psize(2));
    cpatch(ii).patch = crop_im;
    cpatch(ii).label = int32(0);           % negative label
    cpatch(ii).rngkey = randi(RNG_MAX,1,'int32');
  end
end

function neg_bbx = sample_negative(im, psize, N)
% mutiple pos_bbxs
[dimy,dimx,~] = size(im);
stepx = round((psize(1)-1)/4);
stepy = round((psize(2)-1)/4);
sizx = psize(1);
sizy = psize(2);

padpixelx  = max(0, floor((sizx-1)/2)); % margin
padpixely  = max(0, floor((sizy-1)/2)); % margin

x1 = (-padpixelx):stepx:(dimx+padpixelx-sizx+1);
y1 = (-padpixely):stepy:(dimy+padpixely-sizy+1);
x2 = x1 + double(sizx) - 1;
y2 = y1 + double(sizy) - 1;

isvalid = true(numel(y1), numel(x1));

[a,b] = find(isvalid);
neg_ids = 1:numel(a);
%
assert(N < numel(neg_ids));
neg_ids = neg_ids(randperm(numel(neg_ids),N));
% neg_ids = neg_ids(randi(numel(neg_ids),[N,1]));
a = a(neg_ids);
b = b(neg_ids);
neg_bbx = [x1(1,b)', y1(1,a)', x2(1,b)', y2(1,a)'];
