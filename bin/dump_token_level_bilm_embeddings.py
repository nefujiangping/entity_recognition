import os
import h5py
from bilm import TokenBatcher, BidirectionalLanguageModel
import time

import tensorflow as tf
import numpy as np


DTYPE = 'float32'

EXAMPLE_COUNT = 17000


def extract_embedding_weight_file(weight_file, embedd_out_file):
    with h5py.File(weight_file, 'r') as fin, h5py.File(embedd_out_file, 'w') as fout:
        embeddings = fin['embedding']
        fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )


def dump_token_bilm_embeddings(vocab_file, dataset_file, options_file,
                         weight_file, embedding_weight_file, outfile):

    batcher = TokenBatcher(vocab_file)

    ids_placeholder = tf.placeholder('int32', shape=(None, None))

    model = BidirectionalLanguageModel(options_file,
                                       weight_file,
                                       use_character_inputs=False,
                                       embedding_weight_file=embedding_weight_file)
    ops = model(ids_placeholder)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sentence_id = 0
        with open(dataset_file, 'r') as fin, \
                h5py.File(outfile, 'w') as fout:
            for line in fin:
                sentence = line.strip().split()
                token_ids = batcher.batch_sentences([sentence])
                embeddings = sess.run(
                    ops['lm_embeddings'], feed_dict={ids_placeholder: token_ids}
                )
                embedding = embeddings[0, :, :, :]
                ds = fout.create_dataset(
                    '{}'.format(sentence_id),
                    embedding.shape, dtype='float32',
                    data=embedding
                )
                # static_token_emb = embedding[0, :, :]
                # first_layer_emb = embedding[1, :, :]
                # final_layer_emb = embedding[2, :, :]
                # avg_emb = np.mean(embedding, axis=0)  # average embedding of the three layers
                sentence_id += 1
                if sentence_id % 500 == 0:
                    print('%.2f%% finished!' % (sentence_id/float(EXAMPLE_COUNT)*100))


datadir = '/home/xx/workspace/bilm-tf/pretrained'
# the same vocab you used to train the language model
vocab_file = os.path.join(datadir, 'elmo_vocab')
# `options.json` is automatically generated when you trained the language model
options_file = os.path.join(datadir, 'options.json')
# pre-trained model file, which is obtained following the operation described here:
# https://github.com/allenai/bilm-tf#4-convert-the-tensorflow-checkpoint-to-hdf5-for-prediction-with-bilm-or-allennlp
weight_file = os.path.join(datadir, '10epoch_2048dim_256projection_dim_no_charcnn.hdf5')
# this file contains the static token embedding for each token in vocab.
# The file can be extracted from model weight file (i.e. `weight_file` above in this example)
embedding_weight_file = os.path.join(datadir, 'embedding_weight_file.hdf5')

extract_embedding_weight_file(weight_file, embedding_weight_file)

# one sentence per line, Each line is one tokenized sentence (whitespace separated).
dataset_file = '/path/to/data/data.txt'

output_lm_embeddings_file = 'elmo_embeddings.hdf5'

start = time.time()
dump_token_bilm_embeddings(vocab_file, dataset_file, options_file,
                           weight_file, embedding_weight_file,
                           output_lm_embeddings_file)
print(time.time() - start)

with h5py.File(output_lm_embeddings_file, 'r') as fin:
    # second_sentence_embeddings
    key = '1'
    print(fin[key])   # shape (1+`n_layers`, num_tokens, `projection_dim`*2)
    print('static: ')  # if you use the default settings, shape is (3, num_tokens, 1024)
    print(fin[key][0])
    print('first: ')
    print(fin[key][1])
    print('second: ')
    print(fin[key][2])




