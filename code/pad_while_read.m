function [im,iminfo] = pad_while_read(iminfo)

im0 = imreadx(iminfo);

w = size(im0,2);
h = size(im0,1);

if h>w
    pad = floor((h-w)/2);
    padcell_pre = zeros(h,pad,3);
    padcell_after = zeros(h,(h-pad-w),3);
    im = cat(2,padcell_pre,im0,padcell_after);
    iminfo.dim = 2;
    iminfo.pad = pad;
else
    pad = floor((w-h)/2);
    padcell_pre = zeros(pad,w,3);
    padcell_after = zeros((w-pad-h),w,3);
    im = cat(1,padcell_pre,im0,padcell_after);
    iminfo.dim = 1;
    iminfo.pad = pad;
end