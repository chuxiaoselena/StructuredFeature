function [pos_train,pos_val] = LSP_def_idx(name,pos_train,pos_val,mix,pa,cachedir)


posnum = length(pos_train);
valnum = length(pos_val);
cls = ['trainval_def_idx_' name 'M' num2str(mix)];

try
    load([cachedir cls '.mat'],'idx','def');
catch
    data = [pos_train; pos_val];
    def = data_def_lsp(data);
    idx = clusterparts_yy(def,mix,pa);
    save([cachedir cls '.mat'],'idx','def');
end

try
    load([cachedir cls '_SampleBase.mat'],'pos_train','pos_val')
catch
    %----    sample based -------
    all_idx = [];
    for p = 1:length(idx)
        p_idx = idx{p};
        all_idx = [all_idx p_idx];
    end
        
    idx_train = all_idx(1:posnum,:);
    for i = 1:length(pos_train)
        pos_train(i).idx = idx_train(i,:);
    end
    
    idx_val = all_idx(1+posnum:end,:);
    if length(idx_val) ~= valnum
        keyboard;
    end
    for i = 1:length(pos_val)
        pos_val(i).idx = idx_val(i,:);
    end

    save([cachedir cls '_SampleBase.mat'],'pos_train','pos_val','idx_train','idx_val');
end

if length(pos_train) ~= posnum || length(pos_val) ~= valnum
    keyboard;
end