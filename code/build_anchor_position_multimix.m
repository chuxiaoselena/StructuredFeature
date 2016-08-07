function anchor = build_anchor_position_multimix(def,idx,pa,Mix)
% jointmodel = buildmodel(name,model,pa,def,idx,K)
% This function merges together separate part models into a tree structure

globals;
anchor = cell(18,Mix);
for i = 1:length(pa)
    child = i;
    parent = pa(child);
    assert(parent < child);
    
    % add deformation parameter
    p.defid = [];
    if parent > 0
        for k = 1:max(idx{child})
            d.w = [0.01 0 0.01 0];
%             d.i = jointmodel.len + 1;
            x = mean(def{child}(idx{child}==k,1) - def{parent}(idx{child}==k,1));
            y = mean(def{child}(idx{child}==k,2) - def{parent}(idx{child}==k,2));
            anchor{i,k} = round([x+1 y+1]);
        end
    end
end
