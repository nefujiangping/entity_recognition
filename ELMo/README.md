
### Get contextualized sentence representation from pre-trained ELMo model
- Install ELMo scripts: [bilm-tf#installing](https://github.com/allenai/bilm-tf#installing)
- First download pre-trained daguan competition ELMo model to this folder from

  > 链接：https://pan.baidu.com/s/10eZ0EPyerv982aHiEevEgw 提取码：rp5n
  > <br>3 files: `daguan_15epoch_2048dim_256projection_dim_no_charcnn.hdf5`, `elmo_vocab`, `options.json`
  >   + `daguan_15epoch_2048dim_256projection_dim_no_charcnn.hdf5`: obtained by [this step](https://github.com/allenai/bilm-tf#4-convert-the-tensorflow-checkpoint-to-hdf5-for-prediction-with-bilm-or-allennlp)

- Then run `bin/get_elmo_embedding.py`.

- Note that this script will produce very large output files (by default, 6.31G in total).

- Contextualized sentence representations for each sentence of `train_full/dev/test.json` will be dumped to `ELMo_data` folder.
