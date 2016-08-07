//
//  convert_leveldb_to_lmdb.cpp
//
//  Created by Kai Kang on 12/3/15.
//
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <iostream>
#include <sys/stat.h>
#include "boost/scoped_ptr.hpp"
#include "caffe/util/db.hpp"

using namespace caffe;
using std::string;
using boost::scoped_ptr;


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a leveldb to lmdb.\n"
                          "Usage:\n"
                          "    convert_leveldb_to_lmdb INDB SAVEDB\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_leveldb_to_lmdb");
    return 1;
  }

  const char* leveldb_path = argv[1];
  const char* lmdb_path = argv[2];

  // Open databases
  scoped_ptr<db::DB> in_db(db::GetDB("leveldb"));
  scoped_ptr<db::DB> out_db(db::GetDB("lmdb"));
  in_db->Open(leveldb_path, db::READ);
  out_db->Open(lmdb_path, db::WRITE);
  scoped_ptr<db::Cursor> in_iter(in_db->NewCursor());
  scoped_ptr<db::Transaction> txn(out_db->NewTransaction());

  // Copy dataset
  int count = 0;
  for (in_iter->SeekToFirst(); in_iter->valid(); in_iter->Next()) {
    txn->Put(in_iter->key(), in_iter->value());

    if (++count % 1000 == 0) {
      txn->Commit();
      txn.reset(out_db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return 0;
}

