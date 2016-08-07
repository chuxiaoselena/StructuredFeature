function [imdata, tsize] = init_scale(imdata, pa, step)
% also further discover the error in labeling.
len = zeros(length(imdata),length(pa)-1);
invalid = false(length(imdata),length(pa)-1);
for n = 1:length(imdata)
  joints = imdata(n).joints;
  if isfield(imdata(n), 'invalid')
    cur_invalid = imdata(n).invalid;
  end
  for p = 2:size(joints,1)        % p = 1 is root
    len(n,p-1) = norm(abs(joints(p,1:2)-joints(pa(p),1:2)));
    if isfield(imdata(n), 'invalid')
      invalid(n,p-1) = cur_invalid(p) || cur_invalid(pa(p));
      if (~invalid(n,p-1) && len(n,p-1) == 0)
        invalid(n,p-1) = true;
        fprintf( 'Warning: %d part %d->%d seems incorrect, modified as invalid\n', n, p, pa(p) );
        % discover error, update
        imdata(n).invalid(p) = true;
        imdata(n).invalid(pa(p)) = true;
      end
    end
  end
end

r = zeros(1,length(pa)-1);
for i = 1:length(pa)-1
  ratio = log(len(:,i))-log(len(:,1));
  ratio = ratio(~(invalid(:,1) | invalid(:,i)));     % only valid len are used to compute ratio
  assert(~isempty(ratio) && ~any(isnan(ratio)));
  r(i) = exp(median(ratio));
end

scale = zeros(1,length(imdata));
for n = 1:length(imdata)
  norm_len = len(n,:)./r;                        % normalized with respect to the length of [root->its child];
  norm_len = norm_len(~invalid(n,:));            % only valid normalized len are used
  % Node: it fails to estimate the scale of an image, if all the edges are
  % not valid.
  assert(~isempty(norm_len) && ~any(isnan(norm_len)));
  scale(n) = max(18, quantile(norm_len,0.75));   % not designed for images of too low resolution
end

for n = 1:numel(imdata)
  imdata(n).scale_x = scale(n);            % parts are designed not to overlap
  imdata(n).scale_y = scale(n);            % parts are designed not to overlap
end

% ----- get template size that's multiples of step ------
% aspect ratio is kept as 1, so no need to pick mode of aspect ratios
aspect = 1;
% all the part templates currently have the same size, so no need to
% compute template size for every part
w = zeros(1, numel(imdata));
h = zeros(1, numel(imdata));
for n = 1:numel(imdata)
  w(n) = 2 * imdata(n).scale_x + 1;
  h(n) = 2 * imdata(n).scale_y + 1;
end
% pick 5 percentile area
areas = h.*w;
area = quantile(areas, 0.01);
nw = sqrt(area/aspect);
nh = nw*aspect;

tsize = [floor(nh/step), floor(nw/step)];

