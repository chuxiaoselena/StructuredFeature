function imdata = add_rotate(imdata, degree)
if isempty(degree)
  return;
end
num = numel(imdata);
n_degree = numel(degree);
new_imdata = cell(num, 1);
% ---------- roated images --------
parfor ii = 1:num
  im = imreadx(imdata(ii));
  c_imdata = imdata(ii);
  new_imdata{ii} = repmat(c_imdata, [n_degree, 1]);
  for dd = 1:numel(degree)
    new_imdata{ii}(dd).joints = map_rotate_points(c_imdata.joints,im,degree(dd),'ori2new');
    new_imdata{ii}(dd).r_degree = degree(dd);
  end
end

imdata = cat(1, imdata, new_imdata{:});
