#!/usr/bin/env python2
import skimage.io as io
import scipy.io as sio
from caffe_pb2 import Datum, BlobProto
import lmdb
import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert submaps to leveldb.')
    parser.add_argument('submap_dir', type=str,
                        help='Submap directory.')
    parser.add_argument('variable', type=str,
                        help='Variable name in the mat files.')
    parser.add_argument('save_db', type=str,
                        help='Save leveldb.')
    parser.add_argument('mean_proto', type=str,
                        help='Save mean binary proto file.')
    args = parser.parse_args()

    print 'Writing to leveldb {}.'.format(args.save_db)
    db = lmdb.open(args.save_db, max_dbs=2, map_size=1099511627776)
    txn = db.begin(write=True)
    maps = os.listdir(args.submap_dir)
    maps = sorted(maps)

    mean_blob = None
    n = 0
    for cur_map in maps:
        # print "Proccessing {}".format(cur_map)
        key = os.path.splitext(cur_map)[0]

        # try:
        #     value = txn.get(key)
        # except KeyError:
        # Make data blob
        datum = Datum()
        submap = sio.loadmat(os.path.join(args.submap_dir, cur_map))
        submap = submap[args.variable].astype('float')
        if submap.ndim == 3:
            submap = submap.swapaxes(1,2).swapaxes(0,1).astype('float')
            datum.channels, datum.height, datum.width = submap.shape
        else:
            datum.height, datum.width = submap.shape
            datum.channels = 1

        datum.float_data.extend(list(submap.flatten()))
        if mean_blob is None:
            mean_blob = BlobProto()
            mean_blob.height = datum.height
            mean_blob.width = datum.width
            mean_blob.channels = datum.channels
            mean_blob.num = 1
            img_mean = submap
        else:
            img_mean += submap

        datum.label = 0
        if not txn.put(key, datum.SerializeToString(), dupdata=False):
            print 'Key {}: failed.'.format(key)

        n += 1
        if n % 1000 == 0:
            txn.commit()
            print "Proccessed {} samples.".format(n)
            txn = db.begin(write=True)

    # commit last batch
    if n % 1000 != 0:
        txn.commit()
        print "Proccessed {} samples.".format(n)
    img_mean /= len(maps)
    print "Totally proccessed {} samples.".format(n)

    mean_blob.data.extend(list(img_mean.flatten()))
    with open(args.mean_proto, 'wb') as mean_file:
        mean_file.write(mean_blob.SerializeToString())
