function MixGen_lsp(name,pos,lmdb_dir,pa)

wtdir = [lmdb_dir name '/'];
if ~isdir(wtdir)
    mkdir(wtdir);
end
K = 13;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. ov>0.7 positive, ov<0.3 negative, others, not use
% 2. ov>0.7 == dist < scale*0.1   ov<0.2 == dist > scale*0.6
% 3. binary classification
% 5. label: negative 0 positive 1
% 6. mask:  (a) positive on one image, must BP on another one (use 3)
%           (b) sample negative around positive points should be
%               enought (use 2, may or may not use)
%           (c) rest points may use in future (use 1)
%           (d) If you BP one pixel, BP it on all channels!!!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fileID = fopen([lmdb_dir name '_label.txt'],'w');
% dist_map = label_distmap(29); % the value should be odd
% dim3 = size(pos(1).joints,1);
for i = 1:length(pos)
    if mod(i,100)==0
        fprintf(1,'Generate label map: %d/%d\n',i,length(pos));
    end
    if ~isempty(pos(i).joints)  %----- positive sample ------
        subidx = pos(i).idx;
    else %----- negative sample ------
        subidx = ones(1,length(pa));
    end
    subidx = permute(subidx, [1,3,2]);
    save([wtdir sprintf('%06d.mat',i)],'subidx');
end
% fclose(fileID);