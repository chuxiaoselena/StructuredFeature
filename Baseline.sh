GLOG_logtostderr=1 ./caffe-multi/build/tools/caffe train \
-gpu 1 -solver ./protofiles/solver.prototxt \
-weights cache/VGG_ILSVRC_16_layers_full_conv.caffemodel \
2>&1 | tee log_test.txt
