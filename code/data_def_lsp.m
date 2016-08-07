function deffeat = data_def_lsp(pos)
% get absolute positions of parts with respect to HOG cell

% width  = zeros(1,length(pos));
height = zeros(1,length(pos));
points = zeros(size(pos(1).joints,1),size(pos(1).joints,2),length(pos));
for n = 1:length(pos)
    points(:,:,n) = pos(n).joints(:,1:2,:);
end

deffeat = cell(1,size(points,1));
for p = 1:size(points,1)
  def = squeeze(points(p,1:2,:));
  deffeat{p} = (def/8)';
end