./caffe-multi/build/tools/convert_imageset ./ ./external/data/lsp/val.txt \
./cache/lsp/LMDB/LMDB_val_data --encoded

python ./external/data/map2lmdb_datum.py ./external/data/lsp/val_label/ \
map ./cache/lsp/LMDB/LMDB_val_labelmap ./external/data/lsp/mean

python ./external/data/map2lmdb_datum.py ./external/data/lsp/val_mix/ \
subidx ./cache/lsp/LMDB/LMDB_val_mix ./external/data/lsp/mean

./caffe-multi/build/tools/convert_imageset ./ ./external/data/lsp/train.txt \
./cache/lsp/LMDB/LMDB_train_data --encoded

python ./external/data/map2lmdb_datum.py ./external/data/lsp/train_label/ \
map ./cache/lsp/LMDB/LMDB_train_labelmap ./external/data/lsp/mean

python ./external/data/map2lmdb_datum.py ./external/data/lsp/train_mix/ \
subidx ./cache/lsp/LMDB/LMDB_train_mix ./external/data/lsp/mean
