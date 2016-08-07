globals;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();

name = 'lsp';
cachedir = 'cache/lsp/';
lmdb_dir = ['external/data/' name '/'];  % image writing dir
pa = conf.pa;

% -------------------- collect data -----------------------
[pos_train, pos_val,neg_train, ~] = LSP_data_fconv(cachedir);

% -------------------- clustering -----------------------
[pos_train,pos_val] = LSP_def_idx('lsp',pos_train,pos_val,13,pa,cachedir);

% ------------------ generate data and label ------------------------
pos_train = fconv_data_lsp(name,pos_train,pos_val,neg_train,lmdb_dir,cachedir,1);
pos_val = fconv_data_lsp(name,pos_train,pos_val,neg_train,lmdb_dir,cachedir,2);

% ------------------ generate mixturetype ------------------------------
MixGen_lsp('train_mix',pos_train,lmdb_dir,pa);
MixGen_lsp('val_mix',pos_val,lmdb_dir,pa);