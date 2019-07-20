# entity_recognition
Entity recognition codes for "2019 Datagrand Cup: Text Information Extraction Challenge"
## How to train a pure <u>token-level</u> ELMo from scratch?
- Just follow the official instruction described [here](https://github.com/allenai/bilm-tf#training-a-bilm-on-a-new-corpus).
- Some notes:
    + to train a token-level language model, modify `bin/train_elmo.py`: <br/>
    from `vocab = load_vocab(args.vocab_file, 50)` <br/>
    to `vocab = load_vocab(args.vocab_file, None)`
    + modify `n_train_tokens`
    + remove `char_cnn` in `options`
    + modify `lstm.dim`/`projection_dim` as you wish.
    + `n_gpus=2`, `n_train_tokens=94114921`, `lstm['dim']=2048`, `projection_dim=256`, `n_epochs=10`. It took about 17 hours long on 2 GTX 1080 Ti.
- After finishing the [last step](https://github.com/allenai/bilm-tf#4-convert-the-tensorflow-checkpoint-to-hdf5-for-prediction-with-bilm-or-allennlp) of the instruction, you can refer to the [script](https://github.com/nefujiangping/entity_recognition/blob/master/dump_token_level_bilm_embeddings.py) to dump the dynamic sentence representations of your own dataset.
