function cropped = lmdb_fconv_data_lsp(name,pos_train,patchsize,inputsize,cachedir,lmdb_dir)

%----------- crop the original image to training size -----------------
Cropped_dir = [cachedir 'Cropped_'];

try
    load([Cropped_dir name '.mat']);
catch
    wtdir = [lmdb_dir name '/'];
    if ~isdir(wtdir)
        mkdir(wtdir);
    end
    
    fileID = fopen([lmdb_dir name '.txt'],'w');
    for i = 1:length(pos_train)
        if mod(i,100)==0
            fprintf(1,'%d/%d\n',i,length(pos_train));
        end
        
        [im,info_crop] = imtrans_lsp(pos_train(i),patchsize(1),inputsize);  
        cropped(i) = info_crop;
        if 0
            conf = global_conf();
            pa = conf.pa;
            showskeletons_joints(im,info_crop.joints,pa);
        end
        im = uint8(im);
        if size(im,1)~=336 || size(im,2)~=336
            keyboard;
            im = imresize(im,[336,336]);
        end
        imwrite(im,[wtdir sprintf('%06d.png',i)]);
        fprintf(fileID,'%s%06d.png 0\n',wtdir,i);
    end
    fclose(fileID);
    
    save([Cropped_dir name '.mat'],'cropped');
end