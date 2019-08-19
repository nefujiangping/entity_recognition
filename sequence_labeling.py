# coding=utf-8
import json
import os
from tqdm import tqdm

from keras.callbacks import Callback
from keras.layers import *
from keras.models import Model

from keras.optimizers import Adam
from keras_contrib.layers.crf import CRF
from keras.initializers import Constant

from function.data import build_pretrained_embedding, word2id, tag2id, DataGenerator
import config

# word_idx: word_literal, 0: padding, 1: unk
from function.layers import MyBidirectional, Attention
from function.transformation import position_id, dilated_gated_conv1d
from function.metrics import f1_score, recall_score, precision_score

word2vec_path = config.pretrained_embedd
word2vec = build_pretrained_embedding(embedding_file=word2vec_path,
                                      word2id=word2id, embedd_dim=config.embedd_dim, norm=config.norm2one)

maxlen = config.maxlen
# Hyperparameters
batch_size = config.batch_size
emb_dim = config.embedd_dim
hidden_dim = config.hidden_dim
drop_p = config.dropout


def dgcnn_encoder(text_idx, mask_2dim):
    mask = Lambda(lambda x: K.expand_dims(x, 2))(mask_2dim)
    # mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(text_idx)
    # position embedding
    pid = Lambda(position_id)(text_idx)
    position_embedding = Embedding(maxlen, emb_dim, embeddings_initializer='zeros')
    pv = position_embedding(pid)

    # word embedding
    t1 = Embedding(len(word2id)+2, emb_dim, mask_zero=True,
                   embeddings_initializer=Constant(word2vec))(text_idx)  # 0: padding, 1: unk
    t = Add()([t1, pv])  # 字向量、位置向量相加
    t = Dropout(config.dropout)(t)
    t = Lambda(lambda x: x[0] * x[1])([t, mask])
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    if config.DGC12:
        t = dilated_gated_conv1d(t, mask, 1)
        t = dilated_gated_conv1d(t, mask, 2)
        t = dilated_gated_conv1d(t, mask, 5)
        t = dilated_gated_conv1d(t, mask, 1)
        t = dilated_gated_conv1d(t, mask, 1)
        t = dilated_gated_conv1d(t, mask, 1)
    h = Attention(8, 16)([t, t, t, mask])
    h = Concatenate()([t, h])
    sentence_repr = Conv1D(emb_dim, 3, activation='relu', padding='same')(h)
    return sentence_repr


def bilstm_encoder(text_idx, mask_2dim):
    mask_3dim = Lambda(lambda x: K.expand_dims(x, 2))(mask_2dim)  # shape (batch_size, batch_max_seq_len, 1)
    # mask_3dim = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(text_idx)
    embeddings = Embedding(len(word2id)+2, emb_dim,
                           embeddings_initializer=Constant(word2vec))(text_idx)  # 0: padding, 1: unk
    embeddings = Dropout(drop_p)(embeddings)
    embeddings = Lambda(lambda x: x[0]*x[1])([embeddings, mask_3dim])

    output_states = MyBidirectional(CuDNNLSTM(units=int(hidden_dim), return_sequences=True))([embeddings, mask_3dim])
    # LSTM is much slower than CuDNNLSTM, but CuDNNLSTM does't support mask
    # output_states = Bidirectional(LSTM(units=hidden_dim, return_sequences=True))(embeddings)
    sentence_repr = Dropout(drop_p)(output_states)
    return sentence_repr


def entity_recognition_model(encoder='dgcnn', inference='softmax'):
    # Input
    text_idx_in = Input(shape=(None, ))
    tag_label_in = Input(shape=(None, ))
    text_idx, tag_label = text_idx_in, tag_label_in

    # mask_2dim shape(batch_size, seq_len)
    mask_2dim = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(text_idx)
    # mask_3dim = Lambda(lambda x: K.expand_dims(x, 2))(mask_2dim)  # shape (batch_size, batch_max_seq_len, 1)

    # encoder
    if encoder == 'bilstm':
        sentence_repr = bilstm_encoder(text_idx, mask_2dim)
    elif encoder == 'dgcnn':
        sentence_repr = dgcnn_encoder(text_idx, mask_2dim)
    else:
        raise Exception('`encoder` must be `dgcnn` or `bilstm`')

    # inference, loss
    if inference == 'softmax':
        output = Dense(len(tag2id), activation='softmax')(sentence_repr)
        loss = K.sparse_categorical_crossentropy(tag_label, output)
        loss = K.sum(loss * mask_2dim) / K.sum(mask_2dim)
    elif inference == 'crf':
        out = TimeDistributed(Dense(hidden_dim, activation="relu"))(sentence_repr)
        crf = CRF(len(tag2id), sparse_target=True)
        output = crf(out, mask=mask_2dim)
        tag_label = K.one_hot(K.cast(tag_label, 'int32'), len(tag2id))
        loss = crf.get_negative_log_likelihood(y_true=tag_label, X=out, mask=mask_2dim)
    else:
        raise Exception('`inference` must be `softmax` or `crf`')

    pred_model = Model(inputs=[text_idx_in], outputs=[output])
    model = Model(inputs=[text_idx_in, tag_label_in], outputs=[output])
    model.add_loss(loss)

    model.compile(
        optimizer=Adam(lr=1e-3),
    )
    model.summary()
    return model, pred_model


class Evaluate(Callback):

    def __init__(self, model, pred_model, word2id, tag2id, examples, reserved_epochs=[], out_path=None):
        super(Evaluate, self).__init__()
        self.model = model
        self.pred_model = pred_model
        self.word2id = word2id
        self.tag2id = tag2id
        self.id2tag = {idx: tag for tag, idx in self.tag2id.items()}
        self.out_path = out_path
        self.examples = examples
        self.best = 0.
        self.F1 = []
        if not os.path.exists('model/' + config.model_name):
            os.mkdir('model/' + config.model_name)
        self.reserved_epochs = reserved_epochs

    def on_epoch_end(self, epoch, logs=None):
        f1, recall, precision = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best or (epoch in self.reserved_epochs):
            self.best = f1
            self.model.save('model/%s/%d-%s-best_model.weights_f1_%.4f' % (config.model_name, epoch, config.model_name, f1*100))
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best))

    def evaluate(self):
        def pred2tag(pred):
            out = []
            for pred_i in pred:
                out.append([self.id2tag[np.argmax(p)] for p in pred_i])
            return out

        input_text, y_true, y_pred = [], [], []
        for batch in tqdm(DataGenerator(self.examples,
                                        word2id=self.word2id, tag2id=self.tag2id,
                                        batch_size=config.dev_batch_size,
                                        trainset=False).__iter__()):
            batch_text, batch_text_idx, batch_ner_tag = batch[0], batch[1], batch[2]
            predict = self.pred_model.predict(batch_text_idx)  # predict
            if self.out_path:  # if out_path is given, then record input_text for `write_results()`
                input_text.extend(batch_text)
            y_true.extend(batch_ner_tag)  # assemble y_true: label list
            y_pred.extend(pred2tag(predict))  # assemble y_pred: label list
        # y_pred is padded, while y_true is not; recover y_pred to original
        for i in range(len(y_pred)):
            y_pred[i] = y_pred[i][:len(y_true[i])]
        if self.out_path:
            self.write_results(input_text, y_true, y_pred)
        # print(y_true, y_pred)
        f, recall, precision = f1_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)
        return f, recall, precision

    def write_results(self, input_text, y_true, y_pred):
        with open(self.out_path, 'w') as out:
            for sent_text, sent_ground, sent_pred in zip(input_text, y_true, y_pred):
                for word, word_ground, word_pred in zip(sent_text, sent_ground, sent_pred):
                    out.write('%s\t%s\t%s\n' % (word, word_ground, word_pred))
                out.write('\n')


if __name__ == '__main__':
    encoder_mode = 'bilstm'
    inference_mode = 'softmax'
    import sys
    if len(sys.argv) >= 3:
        encoder_mode = sys.argv[1].lower()
        inference_mode = sys.argv[2].lower()
    assert encoder_mode in ['bilstm', 'dgcnn'], '`encoder` must be `dgcnn` or `bilstm`'
    assert inference_mode in ['softmax', 'crf'], '`inference` must be `softmax` or `crf`'

    mode = 'train'
    if mode == 'train':
        train_examples = open(config.train_path, 'r').readlines()
        train_examples = [json.loads(line.strip()) for line in train_examples]
        dev_examples = open(config.dev_path, 'r').readlines()
        dev_examples = [json.loads(line.strip()) for line in dev_examples]

        train_data = DataGenerator(train_examples, batch_size=batch_size, word2id=word2id, tag2id=tag2id)

        model, pred_model = entity_recognition_model(encoder_mode, inference_mode)
        evaluator = Evaluate(model, pred_model, word2id, tag2id, dev_examples)
        model.fit_generator(
            train_data.__iter__(),
            steps_per_epoch=len(train_data),
            epochs=config.epochs,
            callbacks=[evaluator]
        )
    elif mode == 'test':
        model_path = 'baseline-best_model.weights_f1_83.8311'
        model, pred_model = entity_recognition_model(encoder_mode, inference_mode)
        model.load_weights(model_path)
        test_examples = open(config.test_path, 'r').readlines()
        test_examples = [json.loads(line.strip()) for line in test_examples]
        evaluator = Evaluate(model, pred_model, word2id, tag2id, test_examples, out_path='%s-results.txt' % model_path)
        evaluator.evaluate()










