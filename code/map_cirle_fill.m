function mask = map_cirle_fill(mask,joints,lgt_scale,dist_map)

x = joints(2);
y = joints(1);

dist_cen = (size(dist_map,1)-1)/2+1;
%   crop_dist = dist_map( dist_cen - scale_limt: dist_cen + scale_limt, ...
%   dist_cen - scale_limt: dist_cen + scale_limt);
lgt_crop_scale = ceil(lgt_scale);
crop_dist = dist_map( dist_cen - lgt_crop_scale: dist_cen + lgt_crop_scale, ...
    dist_cen - lgt_crop_scale: dist_cen + lgt_crop_scale);

crop = crop_dist;
crop(crop_dist<=lgt_scale) = -1;    % value within thresh
crop(crop_dist>lgt_scale) = 0;       % value outside the thresh

%---------- stick it to label map ------------------
mask(x-lgt_crop_scale: x+lgt_crop_scale, y-lgt_crop_scale:y+lgt_crop_scale) = crop;


