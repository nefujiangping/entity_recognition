""" pretrain a word2vec on the corpus/test/train"""
import argparse
import logging
import os
from os.path import join, exists
from time import time
from datetime import timedelta

import gensim

import re


class Sentences(object):

    def __init__(self, path):
        self._path = path

    def __iter__(self):
        # corpus
        with open(join(self._path, 'corpus.txt'), 'r') as f:
            for line in f:
                if line.strip():
                    yield line.strip().split('_')
        # test
        with open(join(self._path, 'test.txt'), 'r') as f:
            for line in f:
                yield line.strip().split('_')
        # train
        with open(join(self._path, 'train.txt'), 'r') as f:
            for line in f:
                line = re.sub(r'/[abco]\s\s', '_', line)
                yield line.strip()[:-2].split('_')


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time()
    save_dir = args.out_path
    if not exists(save_dir):
        os.makedirs(save_dir)

    sentences = Sentences(args.data_dir)

    model = gensim.models.Word2Vec(
        size=args.dim, window=args.window, min_count=args.min_count, workers=16, sg=1 if args.model=='sg' else 0)
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time()-start)))
    model.train(sentences,
                total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(
        args.dim, len(model.wv.vocab)//1000)))
    model.wv.save_word2vec_format(join(
        save_dir,
        'word2vec.{}.{}d.{}k.min_count{}.window{}.w2v'.format(args.model, args.dim, len(model.wv.vocab)//1000, args.min_count, args.window)
    ))

    print('word2vec trained in {}'.format(timedelta(seconds=time()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--data_dir', required=True, help='root of the training data.')
    parser.add_argument('--out_path', required=True, help='root of the model output, i.e., dir to save word2vec.')
    parser.add_argument('--dim', action='store', type=int, default=128)
    parser.add_argument('--window', action='store', type=int, default=5)
    parser.add_argument('--min_count', action='store', type=int, default=3)
    parser.add_argument('--model', action='store', default='sg', help='sg/cbow, train word2vec in skip-gram or CBOW.')
    args = parser.parse_args()
    assert args.model in ['sg', 'cbow'], '--model must be sg/cbow.'

    main(args)
