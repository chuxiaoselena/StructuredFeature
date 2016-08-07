function dist_map = label_distmap(cache,num)

try
    load([cache 'dist_map' num2str(num) '.mat']);
catch
    dist_map = zeros(num,num);
    cent = (num-1)/2+1;
    for i = 1:num
        for j = 1:num
            dist = sqrt((i-cent)^2+(j-cent)^2);
            dist_map(i,j) = dist;
        end
    end
    save([cache 'dist_map' num2str(num) '.mat'],'dist_map');
end