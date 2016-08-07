function mean_pixel = compute_mean_pixel()
conf = global_conf();
cachedir = conf.cachedir;
mean_pixel_file = [cachedir, 'mean_pixel.mat'];

try
  mean_pixel = parload(mean_pixel_file, 'mean_pixel');
catch
  % compute mean
  caffe_root = conf.caffe_root;
  image_mean_file = [cachedir, 'image_mean.bin'];
  if ~exist(image_mean_file, 'file')
    system(['GLOG_logtostderr=1 ' fullfile(caffe_root, 'build/tools/compute_image_mean'), ' -backend lmdb ', ...
      [cachedir, 'LMDB_train '], image_mean_file]);
    if ~exist(image_mean_file, 'file')
      error('Failed to compute image mean by caffe tool');
    end
  end
  mean_image = caffe('read_mean', image_mean_file);
  mean_pixel = zeros(3,1);
  for ii = 1:3
    mean_pixel(ii) = mean(mean(mean_image(:,:,ii)));
  end
  % round to .01
  mean_pixel = round(mean_pixel*100) / 100;
  parsave(mean_pixel_file, mean_pixel);
end
