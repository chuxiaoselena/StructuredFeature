function pyra = fconv_2_featuremap_MultiMix_lsp(iminfo,net)

% use impyra_exact to control resolution
conf = global_conf();
%------- padding zeros to rectangle -------------
im = pad_while_read(iminfo);

%-------------    image pyramid    ---------------
if (size(im,1) > 600) 
    upS = 0.8;
else
    if (size(im,1) < 336) 
        upS = 368/(min(size(im,1),size(im,2)));
    else
        upS = 1;
    end
end

[pyra,~] = imCNNdet_vgg_multi_mix(im, net, upS);
