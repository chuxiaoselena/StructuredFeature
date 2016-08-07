function Transparent_overlay(im1,im2,trains,maxval)

clear fig
if nargin < 4
  maxval = max(max(max(im2)),1);
end
% im1 = padarray(im1,[0,20],0,'pre');
imagesc(im1);axis image;
% title(til);
hold on;
im2 = imresize(im2,[size(im1,1),size(im1,2)]);
minval = min(min(min(im2)));
% minval =-1;
% maxval = 0.5;

im2 = (im2-minval)*255/(maxval-minval);
H = imagesc(im2);
% title('hh');
set(H, 'AlphaData', trains)
