# coding=utf-8
import keras.backend as K
from keras.layers import Lambda, Conv1D
import numpy as np


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


def seq_meanpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做minpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.mean(seq, 1)


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def get_k_inter(x, n=6):
    seq, k1, k2 = x
    k_inter = [K.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]
    k_inter = [seq_gather([seq, k]) for k in k_inter]
    k_inter = [K.expand_dims(k, 1) for k in k_inter]
    k_inter = K.concatenate(k_inter, 1)
    return k_inter



