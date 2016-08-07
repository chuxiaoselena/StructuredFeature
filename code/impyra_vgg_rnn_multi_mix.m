function pyra = impyra_vgg_rnn_multi_mix(im0, net,upS)
% Compute feature with predefined step
% the step is personal choice


%-------------   image pre-process  ---------------
im = imresize(im0,upS);
im = permute(im,[2,1,3]);       % switch dim1 and dim2
im = im(:,:,3:-1:1);            % RGB to BGR
mean_pixel = 128;
mean_pixel = single(mean_pixel);
mean_pixel = permute(mean_pixel, [3,1,2]);
im = single(im);
im = bsxfun(@minus, im, mean_pixel);

%--------------  get cnn param  ------------------
interval = 10;
% stride = cnnpar.stride;
step = 8;
psize = [224,224];
padx = psize(1)/2; % more than half is visible
pady = psize(2)/2; % more than half is visible
% padx = 0;
% pady = 0;
%--------------  generate multiple image  ------------------
sc = 2 ^(1/interval);
imsize = [size(im, 1), size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/max(psize))/log(sc))-1;
pyra = struct('feat', cell(max_scale,1), 'sizs', cell(max_scale,1), 'scale', cell(max_scale, 1), ...
    'padx', cell(max_scale,1), 'pady', cell(max_scale,1));


scale = zeros(max_scale,1);
sizs = zeros(max_scale,2);

for i = 1:max_scale        % recurrent on pyramid
    scaled = imresize(im, 1/sc^(i-1));

    [dim1,dim2,~] = size(scaled);
    
    net.blobs('data').reshape([dim1 dim2 3 1]);
    scores = net.forward({scaled});
    
    resp = scores{1};
    
    resp = permute(resp,[2,1,3]);
    
    sizs(i,:) = [size(resp,1),size(resp,2)];
    scale(i) = 1/sc^(i-1);
    % ----- pyra info -----
    pyra(i).feat = resp;
    pyra(i).sizs = sizs(i,:);
    pyra(i).scale = step ./ (upS * 1/sc^(i-1));
    pyra(i).pady = pady / step;  % this should be the box-scale on pyra
    pyra(i).padx = padx / step;
end



