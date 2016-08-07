function [pcp_detail,is_matched] = eval_strict_pcp(gt, ests, thresh)
% Evaluate strict PCP
if nargin < 3
  % by default evaluate the PCP@0.5
  thresh = 0.5;
end

assert(numel(ests) == numel(gt));
num = numel(ests);
assert(num >= 1);

% the number of sticks for a pose
stick_n = size(gt(1).sticks, 2);
is_matched = nan(num, stick_n);

for ii = 1:num
  if isempty(ests(ii).sticks)
    fprintf(' WARNING: empty estimation!!\n');
    is_matched(ii, :) = false;
    continue;
  end
  for jj = 1:stick_n
    gt_stick_len = norm( gt(ii).sticks(1:2,jj) - gt(ii).sticks(3:4,jj) );
    is_matched(ii, jj) = norm( ests(ii).sticks(1:2,jj) - gt(ii).sticks(1:2,jj) ) / gt_stick_len <= thresh ...
      && norm( ests(ii).sticks(3:4,jj) - gt(ii).sticks(3:4,jj) ) / gt_stick_len <= thresh;
  end
end

pcp_detail = nan(stick_n, 1);
for jj = 1:stick_n
  pcp_detail(jj) = mean(is_matched(:,jj), 1);
end
