function [accs, range] = eval_pdj(gt, ests, reference_joints_pair, range)
% Evaluate Percentage of Detected Joints (PDJ)
assert(numel(reference_joints_pair) == 2);
assert(numel(ests) == numel(gt));
if nargin < 4
  range = 0:0.01:0.5;
end
num = numel(gt);
assert(num >= 1);
% the number of joints
joint_n = size(gt(1).joints,1);

for ii = 1:num
  gt(ii).scale = norm( gt(ii).joints(reference_joints_pair(1),:) ...
    - gt(ii).joints(reference_joints_pair(2),:) );
end

dists = zeros(num, joint_n);
for ii = 1:num
  if ~isempty(ests(ii).joints)
    dists(ii,:) = sqrt(sum( (ests(ii).joints - gt(ii).joints).^2, 2 ));
    dists(ii,:) = dists(ii,:) / gt(ii).scale;
  else
    fprintf('WARNING: empty estimation\n');
    dists(ii,:) = inf;
  end
end

accs = zeros(numel(range), joint_n);
for ii = 1:numel(range)
  accs(ii,:) = mean(dists <= range(ii),1);
end
