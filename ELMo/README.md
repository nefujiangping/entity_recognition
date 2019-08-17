
### Get contextualized sentence representation from pre-trained ELMo model
- First download pre-trained daguan competition ELMo model from 
链接：https://pan.baidu.com/s/10eZ0EPyerv982aHiEevEgw 提取码：rp5n (3 files: `daguan_15epoch_2048dim_256projection_dim_no_charcnn.hdf5`, `elmo_vocab`, `options.json`).

- Then run `bin/get_elmo_embedding.py`.

- Contextualized sentence representation for `train_full/dev/test.json` will be dumped to `ELMo_data` folder.