Models for Entity Recognition
------
Some **Entity Recognition** models for [2019 Datagrand Cup: Text Information Extraction Challenge](https://www.biendata.com/competition/datagrand/).

## Requirements
- python 3.6
- [keras](https://github.com/keras-team/keras) 2.2.4 (tensorflow backend)
- [keras-contrib](https://github.com/keras-team/keras-contrib) 2.0.8 for CRF inference.
- [gensim](https://pypi.org/project/gensim/) for training word2vec.
- [bilm-tf](https://github.com/allenai/bilm-tf) for ELMo.

## Components of Entity Recognition

### Word Embedding
- Static Word Embedding: word2vec, GloVe
- Contextualized Word Representation: [ELMo](https://github.com/allenai/bilm-tf) (`_elmo`), refer to [Sec.](https://github.com/nefujiangping/entity_recognition#how-to-train-a-pure-token-level-elmo-from-scratch)

### Sentence Representation
- BiLSTM
- [DGCNN](https://kexue.fm/archives/5409)

### Inference
- sequence labeling (`sequence_labeling.py`)
    + CRF
    + softmax
- predict start/end index of entities (`_pointer`)

### Note
According to the three components described above, there actually exists 12 models in all.
However, this repo only implemented the following 6 models:

- Static Word Embedding × (BiLSTM, DGCNN) × (CRF, softmax): `sequence_labeling.py`
- (Static Word Embedding, ELMo) × BiLSTM × pointer: `bilstm_pointer.py` and `bilstm_pointer_elmo.py`

Other models can be implemented by adding/modifying few codes.

## How to run

1. Prepare data:
    1. download official competition data to `data` folder
    1. get sequence tagging train/dev/test data: `bin/trans_data.py`
    1. prepare `vocab`, `tag`
        + `vocab`: word vocabulary, one word per line, with `word word_count` format
        + `tag`: `BIOES` ner tag list, one tag per line (`O` in first line)
    1. follow the step 2 or 3 below
        + 2 is for models using static word embedding
        + 3 is for model using ELMo
1. Run model with static word embedding, here take `word2vec` as an example:
    1. train word2vec: `bin/train_w2v.py`
    1. modify `config.py`
    1. run `python sequence_labeling.py [bilstm/dgcnn] [softmax/crf]` or `python bilstm_pointer.py` (remember to modify `config.model_name` before a new run, or the old model will be overridden)
1. Or run model with ELMo embedding (dump contextualized sentence representation for each sentence of `train/dev/test` to file first, then load them for train/dev/test, **not** run ELMo on the fly):
    1. follow the instruction described [here](ELMo/README.md) to get contextualized sentence representation for `train_full/dev/test` data from pre-trained ELMo weights
    1. modify `config.py`
    1. run `python bilstm_pointer_elmo.py` 


## How to train a pure <u>token-level</u> ELMo from scratch?
- Just follow the official instruction described [here](https://github.com/allenai/bilm-tf#training-a-bilm-on-a-new-corpus).
- Some notes:
    + to train a token-level language model, modify `bin/train_elmo.py`: <br/>
    from `vocab = load_vocab(args.vocab_file, 50)` <br/>
    to `vocab = load_vocab(args.vocab_file, None)`
    + modify `n_train_tokens`
    + remove `char_cnn` in `options`
    + modify `lstm.dim`/`lstm.projection_dim` as you wish.
    + `n_gpus=2`, `n_train_tokens=94114921`, `lstm['dim']=2048`, `projection_dim=256`, `n_epochs=10`. It took about 17 hours long on 2 GTX 1080 Ti.
- After finishing the [last step](https://github.com/allenai/bilm-tf#4-convert-the-tensorflow-checkpoint-to-hdf5-for-prediction-with-bilm-or-allennlp) of the instruction, 
you can refer to the script [dump_token_level_bilm_embeddings.py](bin/dump_token_level_bilm_embeddings.py) to dump the dynamic sentence representations of your own dataset.

## References
- Blog:[《基于CNN的阅读理解式问答模型：DGCNN 》](https://kexue.fm/archives/5409)
- Blog:[《基于DGCNN和概率图的轻量级信息抽取模型 》](https://kexue.fm/archives/6671)
- Named entity recognition tutorial: [Named entity recognition series](https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/)
- Some [codes](https://github.com/bojone/)
- Sequence Evaluation tools: [seqeval](https://github.com/chakki-works/seqeval)
- Neural Sequence Labeling Toolkit: [NCRF++](https://github.com/jiesutd/NCRFpp)
- Contextualized Word Representation: [ELMo](https://github.com/allenai/bilm-tf)
