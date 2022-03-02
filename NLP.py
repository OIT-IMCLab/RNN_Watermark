from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import io
import tensorflow as tf
from keras import regularizers
from keras.losses import categorical_crossentropy
from keras import backend as K
import math

# 埋め込みビット数の指定
bit = 8

# シグモイド関数の定義
def sigmoid(a):
    ans = 1 / (1 + np.exp(-a))

    return ans


def random_index_generator(count):
    indices = np.arange(0, count)
    np.random.shuffle(indices)

    for idx in indices:
        yield idx

# L2正則化
# 適用する場合はlamの値を"0."から変更
def l2_reg(w) :

    lam = 0.
    pe = lam * K.sum(K.square(w))

    return pe

def custom_l2_regularizer(w):

    print("w_shape:" + str(w.get_shape()[0]) + ", " + str(w.get_shape()[1]))
    # 透かしの埋め込み強度
    lam = 0.01
    sbit = str(bit)
    size = w.get_shape()[0] * w.get_shape()[1]
    print(size)

    # 重み係数のベクトル化
    wm_w = K.reshape(w, [size, 1])

    # 重み係数群の個数で埋め込み先を判別
    if not size == 65536:
        rand = "Lk_rand"+ sbit + ".csv"
    else:
        rand = "Lr_rand"+ sbit +".csv"

    if bit == 1:
        key = np.loadtxt(rand).reshape(1,size)
    else:
        key = np.loadtxt(rand).reshape(bit,size)

    key = tf.convert_to_tensor(key, np.float32)


    x_weights = tf.matmul(key, wm_w)
    x_w = tf.sigmoid(x_weights)

    # 透かしの要素を指定
    # condition:奇数ビット, ~condition:偶数ビット
    c = np.arange(0,bit).reshape(bit, 1)
    condition = c % 2 == 0
    c[condition] = 1.
    c[~condition] = 0.

    arr = tf.constant(c, dtype='float32')

    wm_penalty = lam * tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_w,tf.keras.backend.cast_to_floatx(arr)))

    penalty = l2_reg(w)
    loss =  wm_penalty + penalty

    return loss

# Perplexityの定義
def perplexity(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    ppx = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return ppx

def acc(y_true, y_pred):
    c_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(c_pred, tf.float32))
    return accuracy


def watermark_check(w, reg):

    sbit = str(bit)
    size = w.shape[0] * w.shape[1]
    wm_w = K.reshape(w, [size, 1])

    if not size == 65536:
        rand = "Lk_rand"+ sbit + ".csv"
    else:
        rand = "Lr_rand"+ sbit +".csv"

    if bit == 1:
        key = np.loadtxt(rand).reshape(1,size)
    else:
        key = np.zeros(bit, size)
        key = np.loadtxt(rand).reshape(bit,size)

    key = tf.convert_to_tensor(key, np.float32)

    x_w = tf.matmul(key, wm_w)
    x_weights = tf.sigmoid(x_w)

    print(x_weights)

    if reg == "k":
        np.savetxt("kernel_weights.txt", x_weights)
    elif reg == "r":
        np.savetxt("recurrent_weights.txt", x_weights)


# set_seed(0)

path = './data_rojinto_umi.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 8
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()

# 電子透かしの埋め込み先はここで指定する.
# kernel部:kernel_regularizer, recurrent部:recurrent_regularizer
model.add(LSTM(128, input_shape=(maxlen, len(chars)), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[perplexity, acc])
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # 毎回、「老人は老いていた」から文章生成
    for diversity in [0.2]:  # diversity = 0.2 のみとする
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x, y,
                    batch_size=128,
                    epochs=1,
                    callbacks=[print_callback])


layer = model.layers[0]     #summaryよりInput->[0], Dense->[1]なのでmodel.layers[1]
# print(model.layers)
print(layer)

def fig_plot(fit, te):

    title = "model " + te
    label = te + " for training"

    plt.plot(fit.history[te], label=label)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(te)
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


# fig_plot(history, "loss")
# fig_plot(history, "perplexity")
# fig_plot(history, "acc")

def acc_plot(fit, te):

    title = "model " + te
    label = te + " for training"

    plt.plot(fit.history[te], label=label)
    plt.plot(fit.history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(te)
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


acc_plot(history, "accuracy")

# 電子透かしの検出
k_w = layer.get_weights()[0]
r_w = layer.get_weights()[1]

watermark_check(k_w, "k")
watermark_check(r_w, "r")

model.save('LSTM_SentenceGen.h5')
