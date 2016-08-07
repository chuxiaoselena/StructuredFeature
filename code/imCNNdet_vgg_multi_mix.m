function [pyra, unary_map] = imCNNdet_vgg_multi_mix(im, net, upS)

tic;
pyra = impyra_vgg_rnn_multi_mix(im, net, upS);
toc;
% Transparent_overlay(im,pyra(1).feat(:,:,15),0.7);
% Show_all_mix(im,pyra(2).feat(:,:,1:13));
for i = 1:length(pyra)
    unary_map{i} = pyra(i).feat;
    pyra(i).unary = pyra(i).feat;
end
