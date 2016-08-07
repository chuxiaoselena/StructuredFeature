function LabelMapGen_strict(name,pos,labelsize,cache,lmdb_dir)

conf = global_conf;
pa = conf.pa;
wtdir = [lmdb_dir name '_label/'];
if ~isdir(wtdir)
    mkdir(wtdir);
end
dist_map = label_distmap(cache,29); % the value should be odd
dim3 = size(pos(2).joints,1);

if exist([wtdir sprintf('%06d.mat',length(pos))],'file')~=2
    for i = 1:length(pos)
        if mod(i,100)==0
            fprintf(1,'Generate label map: %d/%d\n',i,length(pos));
        end
        if ~isempty(pos(i).joints)  %----- positive sample ------
            data_scale = pos(i).scale_x*pos(i).imrescale/2;
            map = joints2labelmap_lsp_stricter(pos(i),data_scale,labelsize,dist_map);
        else %----- negative sample ------
            map = zeros(labelsize(1),labelsize(2),dim3);
            mask = ones(labelsize(1),labelsize(2),1);
            map = cat(3,map,mask);
            if size(map,1)~=labelsize(1) || size(map,2)~=labelsize(2) || size(map,3) ~= dim3+1
                keyboard;
            end
        end
        if 0
            im = imread([lmdb_dir name sprintf('/%06d.png',i)]);
            subplot(1,2,1);showskeletons_joints(im,round(pos(i).joints),pa);
            subplot(1,2,2);Transparent_overlay(im,map(:,:,end),0.6);
        end
        save([wtdir sprintf('%06d.mat',i)],'map');
    end
end
% fclose(fileID);