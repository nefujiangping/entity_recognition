# coding=utf-8
import json
import os

from keras.callbacks import Callback
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import Constant
import keras.backend as K
from tqdm import tqdm

from function.transformation import seq_and_vec, seq_maxpool
from function.training import ExponentialMovingAverage
from function.layers import MyBidirectional
import config
from function.data import build_pretrained_embedding, word2id, tag2id, sequence_padding
from function.metrics import recall_score, precision_score, f1_score, classification_report

# word_idx: word_literal, 0: padding, 1: unk
word2vec_path = config.pretrained_embedd
word2vec = build_pretrained_embedding(embedding_file=word2vec_path,
                                      word2id=word2id, embedd_dim=config.embedd_dim, norm=config.norm2one)
maxlen = 566
batch_size = config.batch_size
emb_dim = config.embedd_dim
hidden_dim = config.hidden_dim

text_idx_in = Input(shape=(None, ))  # (batch_size, batch_max_seq_len)

# (batch_size, batch_max_seq_len):
# [[0,0,0,1,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0], ...]
a_start_in = Input(shape=(None, ))
# [[0,0,0,0,0,1,0,0,0,1,0,0],[0,0,0,0,1,0,0,0,0,0,0,0], ...]
a_end_in = Input(shape=(None, ))

b_start_in = Input(shape=(None, ))
b_end_in = Input(shape=(None, ))
c_start_in = Input(shape=(None, ))
c_end_in = Input(shape=(None, ))

text_idx, a_start, a_end, b_start, b_end, c_start, c_end = \
    text_idx_in, a_start_in, a_end_in, b_start_in, b_end_in, c_start_in, c_end_in

mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(text_idx)

# word embedding
embeddings = Embedding(len(word2id)+2, emb_dim,
                       embeddings_initializer=Constant(word2vec))(text_idx)  # 0: padding, 1: unk
t = Dropout(config.dropout)(embeddings)
t = Lambda(lambda x: x[0] * x[1])([t, mask])

t = MyBidirectional(CuDNNLSTM(int(hidden_dim/2), return_sequences=True))([t, mask])
t = MyBidirectional(CuDNNLSTM(int(hidden_dim/2), return_sequences=True))([t, mask])

t_max = Lambda(seq_maxpool)([t, mask])
t_dim = K.int_shape(t)[-1]

h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([t, t_max])

h_a = Conv1D(hidden_dim, config.aKernel, activation='relu', padding='same')(h)
h_b = Conv1D(hidden_dim, 3, activation='relu', padding='same')(h)
h_c = Conv1D(hidden_dim, 3, activation='relu', padding='same')(h)
pa_start = Dense(1, activation='sigmoid')(h_a)
pa_end = Dense(1, activation='sigmoid')(h_a)
pb_start = Dense(1, activation='sigmoid')(h_b)
pb_end = Dense(1, activation='sigmoid')(h_b)
pc_start = Dense(1, activation='sigmoid')(h_c)
pc_end = Dense(1, activation='sigmoid')(h_c)

pred_model = Model(inputs=[text_idx_in], outputs=[pa_start, pa_end, pb_start, pb_end, pc_start, pc_end])
model = Model(inputs=[text_idx_in, a_start_in, a_end_in, b_start_in, b_end_in, c_start_in, c_end_in],
              outputs=[pa_start, pa_end, pb_start, pb_end, pc_start, pc_end])


def ptr_loss(ptr_true, ptr_pred, _mask):
    """
    :param ptr_true: shape(batch_size, max_seq_len)
    :param ptr_pred: shape(batch_size, max_seq_len, 1)
    :param _mask: shape(batch_size, max_seq_len, 1)
    :return:
    """
    ptr_true = K.expand_dims(ptr_true, 2)
    _loss = K.binary_crossentropy(ptr_true, ptr_pred)
    _loss = K.sum(_loss * _mask) / K.sum(_mask)
    return _loss


a_start_loss = ptr_loss(a_start, pa_start, mask)
b_start_loss = ptr_loss(b_start, pb_start, mask)
c_start_loss = ptr_loss(c_start, pc_start, mask)
a_end_loss = ptr_loss(a_end, pa_end, mask)
b_end_loss = ptr_loss(b_end, pb_end, mask)
c_end_loss = ptr_loss(c_end, pc_end, mask)
loss = a_start_loss + b_start_loss + c_start_loss + a_end_loss + b_end_loss + c_end_loss

model.add_loss(loss)
model.compile(
    optimizer=Adam(lr=1e-3)
)
model.summary()
fwriter = open('model/%s/model_architecture.txt' % config.model_name, 'w').write
model.summary(print_fn=fwriter)

EMAer = ExponentialMovingAverage(model)
EMAer.inject()


class DataGenerator:

    def __init__(self, data, word2id, tag2id, batch_size, single_pass=True, trainset=True):
        self.trainset = trainset
        self.data = data
        self.word2id = word2id
        self.tag2id = tag2id
        self.batch_size = batch_size
        self.single_pass = single_pass
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def assemble(self, tag_list):
        example_abc_start_end = np.zeros((6, len(tag_list)))  # 0: a_start, 1: a_end
        # 2: b_start, 3: b_end, 4: c_start, 5: c_end
        for idx, tag in enumerate(tag_list):
            if tag == 'B-a':
                example_abc_start_end[0][idx] = 1
            elif tag == 'E-a':
                example_abc_start_end[1][idx] = 1
            elif tag == 'S-a':
                example_abc_start_end[0][idx] = 1
                example_abc_start_end[1][idx] = 1
            elif tag == 'B-b':
                example_abc_start_end[2][idx] = 1
            elif tag == 'E-b':
                example_abc_start_end[3][idx] = 1
            elif tag == 'S-b':
                example_abc_start_end[2][idx] = 1
                example_abc_start_end[3][idx] = 1
            elif tag == 'B-c':
                example_abc_start_end[4][idx] = 1
            elif tag == 'E-c':
                example_abc_start_end[5][idx] = 1
            elif tag == 'S-c':
                example_abc_start_end[4][idx] = 1
                example_abc_start_end[5][idx] = 1
        return example_abc_start_end

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.trainset and not self.single_pass:
                np.random.shuffle(idxs)
            batch_text, batch_text_idx, batch_ner_idx, test_batch_ner = [], [], [], []
            batch_start_end = []
            for i in idxs:
                example = self.data[i]
                example_text = example['text']
                batch_text_idx.append([self.word2id.get(word, 1) for word in example_text])  # 1 for unk
                if self.trainset:  # train mode: append ner idx
                    tag_list = example['ner']
                    example_abc_start_end = self.assemble(tag_list)
                    batch_start_end.append(example_abc_start_end)
                else:  # val/test mode: append ner tag
                    test_batch_ner.append(example['ner'])
                    batch_text.append(example_text)  # val/test mode: append example text
                # assert len(batch_text_idx[-1]) == len(batch_ner_idx[-1]), 'Example length ERROR.'
                if len(batch_text_idx) == self.batch_size or i == idxs[-1]:  # reach the batch_size or the end.
                    if self.trainset:  # train mode: padding ner_idx
                        yield [
                                  sequence_padding(batch_text_idx),  # shape(batch_size, batch_max_seq_len) batch text idx
                                  sequence_padding([batch[0] for batch in batch_start_end]),  # a_start shape(batch_size, batch_max_seq_len)
                                  sequence_padding([batch[1] for batch in batch_start_end]),  # a_end
                                  sequence_padding([batch[2] for batch in batch_start_end]),  # b_start
                                  sequence_padding([batch[3] for batch in batch_start_end]),  # b_end
                                  sequence_padding([batch[4] for batch in batch_start_end]),  # c_start
                                  sequence_padding([batch[5] for batch in batch_start_end]),  # c_end
                              ], None
                        batch_text_idx, batch_start_end = [], []  # empty everything after yielding this batch
                    else:  # eval/test mode: yield ner tag list directly.
                        yield batch_text, \
                              sequence_padding(batch_text_idx), \
                              test_batch_ner
                        batch_text, batch_text_idx, test_batch_ner = [], [], []
            # after finishing one epoch
            if self.single_pass:
                break  # in evaluate or test mode


class Evaluate(Callback):

    def __init__(self, word2id, tag2id, examples, out_path=None):
        super(Evaluate, self).__init__()
        self.word2id = word2id
        self.tag2id = tag2id
        self.out_path = out_path
        self.examples = examples
        self.best = 0.
        self.F1, self.precision, self.recall = [], [], []
        self.passed = 0
        self.stage = 0
        if not os.path.exists('model/' + config.model_name):
            os.mkdir('model/' + config.model_name)

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, recall, precision = self.evaluate()
        self.F1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        if f1 > self.best:
            self.best = f1
            model.save('model/%s/best_model.weights' % config.model_name)
            # model.save('model/%s/%d-%s-best_model.weights_f1_%.4f'
            #            % (config.model_name, epoch, config.model_name, f1*100))
        if epoch == config.epochs-1:  # save latest model
            print('='*10+'LAST'+'='*10)
            model.save('model/%s/%d-best_model.weights' % (config.model_name, epoch))
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
                self.stage == 0 and epoch > 10 and
                (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            model.load_weights('model/%s/best_model.weights' % config.model_name)
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def handle_one_example(self, _a_s, _a_e, _b_s, _b_e, _c_s, _c_e, ex_len):
        tag_seq = ['O']*ex_len

        def handle_one_type(_type, _x_s, _x_e):
            _k1, _k2 = _x_s, _x_e
            _k1, _k2 = _k1[:ex_len], _k2[:ex_len]
            for i, _kk1 in enumerate(_k1):
                if _kk1 > config.start_threshold:
                    for j, _kk2 in enumerate(_k2[i:]):
                        if _kk2 > config.end_threshold:
                            flag = True
                            for test_i in range(i, i+j+1):
                                if tag_seq[test_i] != 'O':
                                    flag = False
                            if flag:  # this span is still available
                                if j == 0:  # Single-word span
                                    tag_seq[i] = 'S-%s' % _type  # S-a/b/c
                                elif j == 1:  # two-word span
                                    tag_seq[i] = 'B-%s' % _type
                                    tag_seq[i+j] = 'E-%s' % _type
                                else:  # three-or-more-word span
                                    tag_seq[i] = 'B-%s' % _type
                                    for I_i in range(i+1, i+j):
                                        tag_seq[I_i] = 'I-%s' % _type
                                    tag_seq[i+j] = 'E-%s' % _type
                                break

        handle_one_type('a', _a_s, _a_e)
        handle_one_type('b', _b_s, _b_e)
        handle_one_type('c', _c_s, _c_e)
        return tag_seq

    def output2tag(self, output, batch_lens):
        a_start, a_end, b_start, b_end, c_start, c_end = output
        num_example = a_start.shape[0]
        ret_batch = []
        for ex_idx in range(num_example):
            ex_len = batch_lens[ex_idx]
            _a_s, _a_e, _b_s, _b_e, _c_s, _c_e = \
                a_start[ex_idx, :, 0], a_end[ex_idx, :, 0], \
                b_start[ex_idx, :, 0], b_end[ex_idx, :, 0], \
                c_start[ex_idx, :, 0], c_end[ex_idx, :, 0],
            pred_tag = self.handle_one_example(_a_s, _a_e, _b_s, _b_e, _c_s, _c_e, ex_len)
            ret_batch.append(pred_tag)
        return ret_batch

    def evaluate(self):
        input_text, y_true, y_pred = [], [], []
        idx = 0
        for batch in tqdm(DataGenerator(self.examples,
                                        word2id=self.word2id, tag2id=self.tag2id,
                                        batch_size=config.dev_batch_size,
                                        trainset=False).__iter__()):
            idx += 1
            batch_text, batch_text_idx, batch_ner_tag = batch[0], batch[1], batch[2]
            batch_lens = [len(ex) for ex in batch_ner_tag]
            predict = pred_model.predict(batch_text_idx)  # predict
            pred_batch_tag = self.output2tag(predict, batch_lens)
            if self.out_path:  # if out_path is given, then record input_text for `write_results()`
                input_text.extend(batch_text)
            if idx == 1:
                print(pred_batch_tag[0])
            y_true.extend(batch_ner_tag)  # assemble y_true: label list
            y_pred.extend(pred_batch_tag)  # assemble y_pred: label list
        if self.out_path:
            self.write_results(input_text, y_true, y_pred)
        # print(y_true)
        # print(y_pred)
        f, recall, precision = f1_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)
        print(classification_report(y_true, y_pred))
        return f, recall, precision

    def write_results(self, input_text, y_true, y_pred):
        with open(self.out_path, 'w') as out:
            for sent_text, sent_ground, sent_pred in zip(input_text, y_true, y_pred):
                for word, word_ground, word_pred in zip(sent_text, sent_ground, sent_pred):
                    out.write('%s\t%s\t%s\n' % (word, word_ground, word_pred))
                out.write('\n')


def write_logs(evaluator, loss, output):
    with open(output, 'w') as out_f:
        for epoch, (_loss, p, r, f) \
                in enumerate(zip(loss,
                                 evaluator.precision,
                                 evaluator.recall,
                                 evaluator.F1)):
            out_f.write('Epoch %d\nloss: %.6f, precision: %.4f, recall: %.4f, f1: %.4f\n' %
                        (epoch, _loss, p, r, f))
        out_f.write('\nbest_f1: %.4f, #epoch: %d' % (evaluator.best, np.argmax(evaluator.F1)))


mode = 'train'
if mode == 'train':
    train_examples = open(config.train_path, 'r').readlines()
    train_examples = [json.loads(line.strip()) for line in train_examples]
    dev_examples = open(config.dev_path, 'r').readlines()
    dev_examples = [json.loads(line.strip()) for line in dev_examples]

    train_data = DataGenerator(data=train_examples,
                               batch_size=batch_size,
                               word2id=word2id, tag2id=tag2id,
                               single_pass=False,
                               trainset=True)
    evaluator = Evaluate(word2id, tag2id, dev_examples)
    history_callback = model.fit_generator(
        train_data.__iter__(),
        steps_per_epoch=len(train_data),
        epochs=config.epochs,
        callbacks=[evaluator]
    )
    # History.history
    loss_history = history_callback.history["loss"]
    write_logs(evaluator, loss_history, 'model/%s/train.log' % config.model_name)
elif mode == 'test':
    model.load_weights('model/%s/best_model.weights' % config.model_name)






