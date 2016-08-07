function [score,Ix,Iy,Im] = passmsg_my_multi2(child,parent,anchor,K)

% INF = 1e10;
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);
[Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));


for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt_yy(child.score(:,:,k), ...
        child.w(1,1), child.w(1,2), child.w(1,3), child.w(1,4),...
        anchor{k}(1),anchor{k}(2),Nx,Ny,1);
%     score0(:,:,k) = score0(:,:,k)+pascore;
end

% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
L  = K;
N  = Nx*Ny;
i0 = reshape(1:N,Ny,Nx);
[score,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));


for l = 1:L
% 	b = child.b(1,l,:);
	[score(:,:,l),I] = max(score0,[],3);
	i = i0 + N*(I-1);
	Ix(:,:,l)    = Ix0(i);
	Iy(:,:,l)    = Iy0(i);
	Im(:,:,l)    = I;
end