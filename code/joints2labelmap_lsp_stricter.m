function [map,iminfo] = joints2labelmap_lsp_stricter(iminfo,gtscale,labelsize,dist_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. label: negative 0 positive 1
% 2. mask: not use 0, use 1
% 3. concat together(label, map)
% 4.  ov>0.7 == dist < scale*0.1   ov<0.2 == dist > scale*0.6
% 5 !!!! mask:  (a) positive on one map, must BP on another one (use 3)
%               (b) sample negative around positive points should be
%               enought (use 2, may or may not use)
%               (c) rest points may use in future (use 1)
%               (d) If you BP one pixel, BP it on all channels!!!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim1 = labelsize(1);
dim2 = labelsize(2);

% mask = zeros(dim1,dim2);
joints = iminfo.joints;
label_joints = round((joints+4)/8);

pad = 10;
map = zeros(dim1+pad*2,dim2+pad*2,size(label_joints,1)); % pading zeros around
label_joints = label_joints+pad; % shift label paded pixels
lgt_scale = floor(gtscale/8)*0.2;

% valid = iminfo.negBP;

% if valid
mask = ones(dim1+pad*2,dim2+pad*2,size(label_joints,1));   % init as 1
% else
%      mask = zeros(dim1+pad*2,dim2+pad*2,size(label_joints,1));   % init as 0
% end

%------------- find regions close to gt(set them as -1, not use) ---------
lgt_scale = floor(gtscale/8*0.4);
for i = 1:size(label_joints,1)
    mask(:,:,i) = map_cirle_fill(mask(:,:,i),label_joints(i,:),lgt_scale,dist_map);
end

% icon_mask = zeros(dim1+pad*2,dim2+pad*2);
icon_mask = min(mask,[],3);
icon_mask_sig = icon_mask;
icon_mask(icon_mask_sig<0) = 1;
icon_mask(icon_mask_sig>=0) = 0;
if size(icon_mask,1) ~= size(mask,1)
    keyboard;
end

%------------- find regions around, set as 2 ---------
scale_limt = ceil(gtscale/8*0.5);
for i = 1:size(label_joints,1)
    rawmask = map_cirle_fill_around(mask(:,:,i),label_joints(i,:),lgt_scale,scale_limt,dist_map,1);
    rawmask(icon_mask==1) = 0;
    mask(:,:,i) = rawmask;
end


for i = 1:size(label_joints,1)
    map(label_joints(i,2),label_joints(i,1),i) = 1;
    mask(label_joints(i,2),label_joints(i,1),:) = 3;  % if this pixel is used, use it for all class
    %     if lgt_scale>1
    %         map(label_joints(i,2)+1,label_joints(i,1),i) = 1;   %up
    %         map(label_joints(i,2)-1,label_joints(i,1),i) =1;   %down
    %         map(label_joints(i,2),label_joints(i,1)-1,i) = 1;   %left
    %         map(label_joints(i,2),label_joints(i,1)+1,i) = 1;   %right
    %         %----------- (a). positive on any must BP on the rest ------------
    %         mask(label_joints(i,2)+1,label_joints(i,1),:) = 3;   %up
    %         mask(label_joints(i,2)-1,label_joints(i,1),:) = 3;   %down
    %         mask(label_joints(i,2),label_joints(i,1)-1,:) = 3;   %left
    %         mask(label_joints(i,2),label_joints(i,1)+1,:) = 3;   %right
    %     end
end

mask = max(mask,[],3);

%------------------- (a). BP pixels around  --------------------

map = map(pad+1:end-pad,pad+1:end-pad,:);
mask = mask(pad+1:end-pad,pad+1:end-pad,:);

map =cat(3,map,mask);
if size(map,1)~=labelsize(1) || size(map,2)~=labelsize(2) || size(map,3) ~= size(label_joints,1)+1
    keyboard;
end



