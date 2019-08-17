# Convention of `model_name`:
# `model_name` with suffix '_full' means train on the train_full(train+dev) set,
#  while without suffix '_full' means train on the train split.
model_name = '2bi-crf'  # _full

aKernel = 3  # Kernel size for Conv1D for `span a`
DGC12 = False  # Use 6-layer DGC or 12-layer

start_threshold, end_threshold = 0.5, 0.4  # span start/end index threshold for `_pointer` models.

# elmo
# `avg/last`, `avg` means use the embedding of the average of three layers of ELMo
# `last` means use the last layer.
elmo_embedding_mode = 'avg'
elmo_dim = 512

# Contextualized sentence embedding of ELMo.
# Please refer to Sec. `How to train a pure token-level ELMo from scratch` of `README.md`
# train_elmo_path = 'daguan_data_elmo_embedd/train_elmo_embeddings_%s.hdf5' % elmo_embedding_mode
# dev_elmo_path = 'daguan_data_elmo_embedd/dev_elmo_embeddings_%s.hdf5' % elmo_embedding_mode
# test_elmo_path = 'daguan_data_elmo_embedd/test_elmo_embeddings_%s.hdf5' % elmo_embedding_mode
train_elmo_path = '/home2/public/jp/daguan_data_elmo_embedd/train_elmo_embeddings_%s.hdf5' % elmo_embedding_mode
dev_elmo_path = '/home2/public/jp/daguan_data_elmo_embedd/dev_elmo_embeddings_%s.hdf5' % elmo_embedding_mode
test_elmo_path = '/home2/public/jp/daguan_data_elmo_embedd/test_elmo_embeddings_%s.hdf5' % elmo_embedding_mode

# data for train/dev/test
train_path = 'data/train%s.json' % ('_full' if 'full' in model_name else '')
dev_path = 'data/dev.json'
test_path = 'data/test.json'

maxlen = 566  # maximum of sequence length

# word vector such as word2vec, GloVE
# data format: one word per line. embedd_dim+1 tokens split by space.
# The first token is the word itself, and the rest is the corresponding word vector.
pretrained_embedd = 'data/word2vec.min3.128d.12k.w2v'
norm2one = True

vocab_path = 'data/vocab.4686'  # vocabulary used to train/test. one word per line, format `word word_count`
tag_path = 'data/tag'  # tag vocab. 'O' must be in the first line

# Hyperparameters
embedd_dim = 128
hidden_dim = 256
batch_size = 50
dev_batch_size = 50

dropout = 0.25
epochs = 100




