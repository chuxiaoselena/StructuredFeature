function idx = clusterparts_yy(deffeat,K,pa)


R = 50;
K = ones(1,length(pa))*K;
idx = cell(1,length(deffeat));
for p = 1:length(deffeat)
    % create clustering feature
    if pa(p) == 0
        i = 1;
        while pa(i) ~= p
            i = i+1;
        end
        X = deffeat{i} - deffeat{p};
    else
        X = deffeat{p} - deffeat{pa(p)};
    end
    % try multiple times kmeans
    gInd = cell(1,R);
    cen  = cell(1,R);
    sumdist = zeros(1,R);
    fprintf('Clustering Class: %d \n',p);
    for trial = 1:R
        if mod(trial,10)==0
            fprintf('     trial: %d \n',trial);
        end
        [gInd{trial} cen{trial} sumdist(trial)] = k_means(X,K(p));
    end
    % take the smallest distance one
    [dummy ind] = min(sumdist);
    idx{p} = gInd{ind(1)};
end






