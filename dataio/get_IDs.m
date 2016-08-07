function [nbh_IDs, global_IDs, target_IDs] = get_IDs(pa, K)
p_no = length(pa);
nbh_IDs = cell(p_no, 1);
target_IDs = cell(p_no, 1);
t_cnt = 1;
for ii = 1:p_no
  for jj = 1:p_no
    if pa(ii) == jj || pa(jj) == ii
      nbh_IDs{ii} = cat(1, nbh_IDs{ii}, jj);
    end
  end
  target_IDs{ii} = t_cnt:t_cnt + numel(nbh_IDs{ii})-1;
  t_cnt = t_cnt + numel(nbh_IDs{ii});
end

% ------ global_IDs ---------
global_IDs = cell(p_no, 1);
it = 0;
for p = 1:p_no
  nbh_N = numel(nbh_IDs{p});
  ks = zeros(nbh_N, 1);
  for n = 1:nbh_N
    ks(n) = K;
  end
  global_IDs{p} = reshape(it + (1:prod(ks)), cat(1,ks,1)');
  it = it + numel(global_IDs{p});
end
