import json
import os
import h5py
from bilm import TokenBatcher, BidirectionalLanguageModel
import time

import tensorflow as tf
import numpy as np


DTYPE = 'float32'

EXAMPLE_COUNT = 1500

datadir = 'ELMo'
vocab_file = os.path.join(datadir, 'elmo_vocab')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'daguan_15epoch_2048dim_256projection_dim_no_charcnn.hdf5')
inp_embedding_weight_file = os.path.join(datadir, 'embedding_weight_file')
ELMo_data = 'ELMo_data'


def extract_embedding_weight_file(weight_file, embedd_out_file):
    with h5py.File(weight_file, 'r') as fin, h5py.File(embedd_out_file, 'w') as fout:
        embeddings = fin['embedding']
        fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )


def dump_bilm_embeddings(vocab_file, options_file,
                         weight_file, embedding_file):

    batcher = TokenBatcher(vocab_file)

    ids_placeholder = tf.placeholder('int32', shape=(None, None))
    model = BidirectionalLanguageModel(options_file,
                                       weight_file,
                                       use_character_inputs=False,
                                       embedding_weight_file=embedding_file)
    ops = model(ids_placeholder)

    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True

    def dump_split_bl_embeddings(_sess, dataset_file, outfile_last, outfile_avg, outfile_avglasttwo):
        if outfile_avglasttwo:
            fout_avglasttwo = h5py.File(outfile_avglasttwo, 'w')
        if outfile_last:
            fout_last = h5py.File(outfile_last, 'w')
        if outfile_avg:
            fout_avg = h5py.File(outfile_avg, 'w')
        fin = open(dataset_file, 'r')
        try:
            sentence_id = 0
            for line in fin:
                sentence = json.loads(line.strip())['text']
                # sentence = line.strip().split()
                _ids = batcher.batch_sentences([sentence])
                embeddings = sess.run(
                    ops['lm_embeddings'], feed_dict={ids_placeholder: _ids}
                )
                embedding = embeddings[0, :, :, :]
                if outfile_last:
                    last_layer_emb = embedding[2, :, :]
                    last_ds = fout_last.create_dataset(
                        '{}'.format(sentence_id),
                        last_layer_emb.shape, dtype='float32',
                        data=last_layer_emb
                    )
                if outfile_avg:
                    avg_emb = np.mean(embedding, axis=0)
                    avg_ds = fout_avg.create_dataset(
                        '{}'.format(sentence_id),
                        avg_emb.shape, dtype='float32',
                        data=avg_emb
                    )
                if outfile_avglasttwo:
                    avg_lasttwo = np.mean(embedding[1:3, :, :], axis=0)
                    avglasttwo_ds = fout_avglasttwo.create_dataset(
                        '{}'.format(sentence_id),
                        avg_lasttwo.shape, dtype='float32',
                        data=avg_lasttwo
                    )
                sentence_id += 1
                if sentence_id % 500 == 0:
                    print('%.2f%% finished!' % (sentence_id/float(EXAMPLE_COUNT)*100))
        finally:
            fin.close()
            if outfile_avglasttwo:
                fout_avglasttwo.close()
            if outfile_last:
                fout_last.close()
            if outfile_avg:
                fout_avg.close()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for _count, split in zip([1500, 17000, 3000], ['dev', 'train_full', 'test']):
            # split = 'dev'  # dev/train_full/test
            dataset_file = 'data/%s.json' % split
            output_file_last = os.path.join(ELMo_data, '%s_elmo_embeddings_last.hdf5' % split)
            output_file_avg = os.path.join(ELMo_data, '%s_elmo_embeddings_avg.hdf5' % split)
            output_file_avg_of_last_two = os.path.join(ELMo_data, '%s_elmo_embeddings_avg_of_last_two.hdf5' % split)
            EXAMPLE_COUNT = _count
            print('start to dump %s split...' % split)
            start = time.time()
            dump_split_bl_embeddings(sess,
                                     dataset_file,
                                     output_file_last,
                                     output_file_avg,
                                     output_file_avg_of_last_two)
            print('%.1f mins.' % ((time.time() - start)/60.))


# get `embedding_weight_file`
extract_embedding_weight_file(weight_file, inp_embedding_weight_file)

if not os.path.exists(ELMo_data):
    os.mkdir(ELMo_data)
dump_bilm_embeddings(vocab_file, options_file,
                     weight_file, inp_embedding_weight_file)




