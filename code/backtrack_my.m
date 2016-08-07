function box = backtrack_my(x,y,mix,parts,pyra)
numx     = length(x);
numparts = length(parts);

xptr = zeros(numx,numparts);
yptr = zeros(numx,numparts);
mptr = zeros(numx,numparts);
box  = zeros(numx,4,numparts);
% joints = zeros(numparts,2);
for k = 1:numparts,
    p   = parts(k);
    if k == 1,
        xptr(:,k) = x;
        yptr(:,k) = y;
        mptr(:,k) = mix;
    else
        % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
        par = p.parent;
        [h,w,foo] = size(p.Ix);
        I   = (1-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
%         I   = (mptr(:,par)-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
        xptr(:,k) = p.Ix(I);
        yptr(:,k) = p.Iy(I);
        mptr(:,k) = 1;
%         mptr(:,k) = p.Im(I);
    end
    scale = pyra.scale;
    x = (xptr(:,k) - 1)*scale+1;
    y = (yptr(:,k) - 1)*scale+1;
%     joints(k,:,:) = [x(1),y(1)];
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + pyra.padx*2*scale - 1;
    y2 = y1 + pyra.pady*2*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
%     box(:,:,k) = [x1 y1 x2 y2 mptr(:,k)];
end
% 
