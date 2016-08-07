function det = lsp_box2det(boxes, p_no)
switch p_no
  case 26
    I = [1  2  3  4  5  6  7  8  9  10 11 12 13 14];
    J = [1  2  3  5  7 10 12 14  15 17 19 22  24 26];
    A = [1  1  1  1  1  1  1  1  1  1  1  1  1  1];
    Transback = full(sparse(I,J,A,14,26));
  case 14
    Transback = eye(14,14);
end


det = struct('joints', cell(1, numel(boxes)), 'score', cell(1, numel(boxes)));
for n = 1:length(boxes)
  if isempty(boxes{n})
    continue
  end
  box = boxes{n};
  b = box(:, 1:(p_no*4));
  b = reshape(b, size(b,1), 4, p_no);
  b = permute(b,[1 3 2]);
  bx = .5*b(:,:,1) + .5*b(:,:,3);
  by = .5*b(:,:,2) + .5*b(:,:,4);
  for i = 1:size(b,1)
    det(n).joints(:,:,i) = Transback * [bx(i,:)' by(i,:)'];
    det(n).score(i) = box(i, end);
  end
end