function boxes = test_mix_postprocess_lsp(name,net,test)

conf = global_conf;
pa = conf.pa;

cachedir = ['cache/TestModel_mix_lsp/postp_' name '/'];
if ~isdir(cachedir), mkdir(cachedir); end
single_feature = [cachedir 'Single_feature/'];
if ~isdir(single_feature), mkdir(single_feature); end
single_feature = [cachedir 'Single_feature/'];
if ~isdir(single_feature), mkdir(single_feature); end




try
    load([cachedir name '_rawbox.mat'])
catch
    % --------------------- Extract features --------------------------
    for i = 1:length(test)
        if exist([single_feature sprintf('pyra_%04d.mat',i)],'file')~=2
            fprintf([name ': testing: %d/%d\n'],i,length(test));
            pyra = fconv_2_featuremap_MultiMix_lsp(test(i),net);
            save([single_feature sprintf('pyra_%04d.mat',i)],'pyra');
        end
    end
    
    % ----------------------- Post-process --------------------------
    try
        load([cachedir 'anchor_multi.mat'])
    catch
        load('cache/lsp/trainval_def_idx_lspM13.mat');
%         load('cache/lsp/trainval_idx_def.mat'); uncomment this line if
%         you are going to use the model we provided
        anchor = build_anchor_position_multimix(def,idx,pa,13);
        save([cachedir 'anchor_multi.mat'],'anchor');
    end
    
    
    boxes = cell(1,length(test));
    for i = 1:length(boxes)
        fprintf([name ': process: %d/%d\n'],i,length(test));
        load([single_feature sprintf('pyra_%04d.mat',i)]);
        [~,info] = pad_while_read(test(i));
        box = post_multimix_dt_lsp(info,pyra,anchor);
        boxes{i} = nms_pose(box,0.3);
        
        if 0
            im = imreadx(test(i));
            showskeletons(im,boxes{i},pa);
        end
    end
    
    for i = 1:length(boxes)
        subbox =  boxes{i};
        subbox = reshape(subbox(1:end-1),4,[]);
        subbox = subbox(:)';
        boxes{i} = [subbox boxes{i}(end)];
    end
    
    save([cachedir name '_rawbox.mat'],'boxes');
end

