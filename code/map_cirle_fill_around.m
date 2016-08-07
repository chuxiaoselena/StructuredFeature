function mask = map_cirle_fill_around(mask,joints,lgt_scale,scale_limt,dist_map,valid)

x = joints(2);
y = joints(1);

dist_cen = (size(dist_map,1)-1)/2+1;
crop_dist = dist_map( dist_cen - scale_limt: dist_cen + scale_limt, ...
    dist_cen - scale_limt: dist_cen + scale_limt);


% crop_dist = dist_map( dist_cen - lgt_scale: dist_cen + lgt_scale, ...
%     dist_cen - lgt_scale: dist_cen + lgt_scale);


crop = crop_dist;
crop(crop_dist<=lgt_scale) = 0;
crop(crop_dist>lgt_scale) = 2;    % value outside inner thresh

if valid                             % value outside outer thresh
    crop(crop_dist>scale_limt) = 1;
else
    crop(crop_dist>scale_limt) = 0;
end

%---------- stick it to label map ------------------
% x1 = max(1,x-scale_limt);
% x2 = min(size(mask,1),x+scale_limt);
% y1 = max(1,y-scale_limt);
% y2 = min(size(mask,2),y+scale_limt);
% mask(x1: x2, y1:y2) = crop;
mask(x-scale_limt: x+scale_limt, y-scale_limt:y+scale_limt) = crop;
%  mask(x-lgt_scale: x+lgt_scale, y-lgt_scale:y+lgt_scale) = crop;


