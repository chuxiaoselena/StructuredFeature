function imdata = add_flip(imdata, mirror)
% ---------- flip version --------


num = numel(imdata);
new_imdata = cell(num, 1);

for ii = 1:num
  c_imdata = imdata(ii);
  new_imdata{ii} = c_imdata;
  if isfield(c_imdata, 'r_degree')
    % never flip rotated images
    assert(c_imdata.r_degree == 0);
  end
  if isfield(c_imdata, 'joints') && ~isempty(c_imdata.joints)
    im = imreadx(c_imdata);
    width = size(im,2);
    new_imdata{ii}.joints(mirror, 1) = width - c_imdata.joints(:,1) + 1;
    new_imdata{ii}.joints(mirror, 2) = c_imdata.joints(:,2);
    % also flip invalid (if exist)
    if isfield(c_imdata, 'invalid')
      new_imdata{ii}.invalid(mirror) = c_imdata.invalid;
    end
  end
  new_imdata{ii}.isflip = 1;
end

imdata = cat(1, imdata, new_imdata{:});
