function boxes = post_multimix_dt_lsp(iminfo,pyra,anchor)
INF = 1e10;
conf = global_conf();
pa = conf.pa;
p_no = 26;
K = 13;
w = [0.03, 0, 0.03, 0];
box = [];
boxes =[];
cnt = 0;
% boxes = zeros(length(pyra),p_no*4+1);
for l = 1:length(pyra)

    feat = pyra(l).feat(:,:,1:end-1);
    sizs= pyra(l).sizs;
    for i = 1:p_no
        parts(i).score = feat(:,:,(i-1)*K+1: i*K);
    end
    
    %-------- pass message -------------
    for i =  p_no:-1:2
        child.score = double(parts(i).score);
        child.w = w;
        parent.score = parts(pa(i)).score;
        [msg,Ix,Iy,Im] = passmsg_my_multi2(child,parent,anchor(i,:),size(anchor,2));
        
        parts(i).Ix = Ix;
        parts(i).Iy = Iy;
        parts(i).Im = Im;
        parts(i).parent = pa(i);
        parts(pa(i)).score = parts(pa(i)).score+msg;
    end
    
    [rscore Im] = max(parts(1).score,[],3);
    
    thresh = max(max(rscore));
    [Y,X] = find(rscore >= thresh);
    
    for i = 1:length(X)
        cnt = cnt + 1;
        x = X(i);
        y = Y(i);
        m = Im(y,x);
        box = backtrack_my(x,y,m,parts,pyra(l));
        pad_pre = iminfo.pad;
        if iminfo.dim ==1
            box(:,2,:) = box(:,2,:)-pad_pre;
            box(:,4,:) = box(:,4,:)-pad_pre;
        else
            box(:,1,:) = box(:,1,:)-pad_pre;
            box(:,3,:) = box(:,3,:)-pad_pre;
        end
        box = reshape(box,1,4*p_no);
        boxes(cnt,:) = [box rscore(y,x)];
    end
    
end



