import numpy as np
import codecs
import config


vocab_items = open(config.vocab_path, 'r').readlines()
tag_items = open(config.tag_path, 'r').readlines()  # 'O' must be in the first line

# word_idx: word_literal, 0: padding, 1: unk
id2word = {idx+2: vocab_items[idx].strip().split()[0] for idx in range(len(vocab_items))}
word2id = {word: idx for idx, word in id2word.items()}

id2tag = {idx: tag_items[idx].strip() for idx in range(len(tag_items))}  # tag_idx: tag_literal, 0: padding
tag2id = {tag: idx for idx, tag in id2tag.items()}


def padding_elmo_embedding(_batch_embedding, elmo_embedd_dim=512):
    maxL = max([len(x) for x in _batch_embedding])
    # print(maxL)
    return np.array([
        np.concatenate([sent, np.zeros((maxL-len(sent), elmo_embedd_dim))])
        if len(sent) < maxL
        else sent
        for sent in _batch_embedding
    ])


def sequence_padding(X, padding=0):
    max_len = max([len(x) for x in X])
    return np.array([
        np.concatenate([x, [padding]*(max_len-len(x))]) if len(x) < max_len else x for x in X
    ])


class DataGenerator:

    def __init__(self, data, word2id, tag2id, batch_size, trainset=True):
        self.trainset = trainset
        self.data = data
        self.word2id = word2id
        self.tag2id = tag2id
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.trainset:
                np.random.shuffle(idxs)
            batch_text, batch_text_idx, batch_ner_idx, test_batch_ner = [], [], [], []
            for i in idxs:
                example = self.data[i]
                example_text = example['text']
                batch_text_idx.append([self.word2id.get(word, 1) for word in example_text])  # 1 for unk
                if self.trainset:  # train mode: append ner idx
                    batch_ner_idx.append([self.tag2id[word] for word in example['ner']])
                else:  # val/test mode: append ner tag
                    test_batch_ner.append(example['ner'])
                    batch_text.append(example_text)  # val/test mode: append example text
                # assert len(batch_text_idx[-1]) == len(batch_ner_idx[-1]), 'Example length ERROR.'
                if len(batch_text_idx) == self.batch_size or i == idxs[-1]:
                    batch_text_idx = sequence_padding(batch_text_idx)
                    if self.trainset:  # train mode: padding ner_idx, expand_dims for training and CRF `sparse_target`
                        batch_ner_idx = sequence_padding(batch_ner_idx)
                        # batch_ner_idx = np.expand_dims(batch_ner_idx, 2)
                        yield [batch_text_idx, batch_ner_idx], None
                        batch_text_idx, batch_ner_idx = [], []
                    else:  # eval/test mode: yield ner tag list directly.
                        yield batch_text, batch_text_idx, test_batch_ner
                        batch_text, batch_text_idx, test_batch_ner = [], [], []
            # after finishing one epoch
            if not self.trainset:
                break  # in evaluate or test mode


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_file, embedd_dim):
    """
        Convert embedding_file to embedding_dict.
        embedding_file format: one word per line.
            n+1 tokens per line. The first token is the word itself, the others consist word_vector
        embedding_dict: `key` is the word (string); `value` is the word vector (numpy.ndarray).
    :param embedding_file: pre-trained embedding file path.
    :param embedd_dim: word embedding dimension.
    :return: embedd_dict
    """
    embedd_dict = dict()
    with codecs.open(embedding_file, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim + 1 != len(tokens):
                continue
            # assert embedd_dim + 1 == len(tokens), 'The given embedding_file\'s embedd_dim is not consistent to the param. Check embedding file/param'
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            word = tokens[0]
            if word in ['lrb', 'rrb', 'rsb', 'lsb']:
                word = '-%s-' % word  # glove: lrb  <===>  vocab: -lrb-
            embedd_dict[word] = embedd
    return embedd_dict


def build_pretrained_embedding(embedding_file, word2id, embedd_dim=128, norm=True):
    """
        reference `https://github.com/jiesutd/NCRFpp/blob/master/utils/functions.py#L162`
        Build pre-trained embedding using pre-trained embedding file (like GloVe) and vocab (containing the word & index mapping).
    :param embedding_file: string, Pre-trained embedding file path.
    :param word2id: dict, key is word, and value is the corresponding index
    :param embedd_dim: int, Embedding dimension.
    :param norm: bool, Norm the word embedding or not.
    :return: numpy.ndarray, pre-trained word embedding. size (vocab.size(), embedd_dim).
    """
    embedd_dict = dict()
    if embedding_file:
        embedd_dict = load_pretrain_emb(embedding_file, embedd_dim)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([len(word2id) + 2, embedd_dim])  # 0 for padding, 1 for unk
    perfect_match, case_match, not_match = 0, 0, 0  # to record match count
    for word, index in word2id.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"
            % (pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/len(word2id)))
    return pretrain_emb

