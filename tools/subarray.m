function B = subarray(A, i1, i2, j1, j2, pad)

% B = subarray(A, i1, i2, j1, j2, pad)
% Extract subarray from array
% pad with boundary values if pad = 1
% pad with zeros if pad = 0

dim = size(A);
B = zeros(i2-i1+1, j2-j1+1, dim(3), 'like', A);
if pad
  B((i1:i2)-i1+1, (j1:j2)-j1+1, :) = A(min(max(i1:i2, 1), dim(1)), min(max(j1:j2,1), dim(2)), :);
else
  ai1 = max(i1, 1);
  ai2 = min(i2, dim(1));
  aj1 = max(j1, 1);
  aj2 = min(j2, dim(2));
  
  B((ai1:ai2)-i1+1, (aj1:aj2)-j1+1, :) = A(ai1:ai2, aj1:aj2, :);
end

