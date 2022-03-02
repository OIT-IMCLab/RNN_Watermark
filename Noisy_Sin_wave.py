from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras import backend as K

# 埋め込みbit数の指定
bit = 8

# loss
def plot_history_loss(fit):
    plt.plot(fit.history['loss'],label="loss for training")
    plt.plot(fit.history['val_loss'],label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

# L2正則化
# 適用する場合にはlamの値を"0."から変更
def l2_reg(w) :

    lam = 0.
    pe = lam * K.sum(K.square(w))

    return pe

def custom_l2_regularizer(w):

    print("w_shape:" + str(w.get_shape()[0]) + ", " + str(w.get_shape()[1]))
    # 透かし埋め込みの強度
    lam = 0.01
    sbit = str(bit)
    size = w.get_shape()[0] * w.get_shape()[1]
    print(size)

    # 重み係数のベクトル化
    wm_w = K.reshape(w, [-1, 1])

    # 重み係数群の個数で埋め込み先を判別
    if not size == 360000:
        rand = "Lk_rand"+ sbit + ".csv"
    else:
        rand = "Lr_rand"+ sbit +".csv"

    if bit == 1:
        key = np.loadtxt(rand).reshape(1,size)
    else:
        key = np.loadtxt(rand).reshape(bit,size)

    key = tf.convert_to_tensor(key, np.float32)

    print(rand)

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


def watermark_check(w, reg):

    sbit = str(bit)
    size = w.shape[0] * w.shape[1]
    wm_w = K.reshape(w, [-1, 1])

    if not size == 360000:
        rand = "Lk_rand"+ sbit + ".csv"
    else:
        rand = "Lr_rand"+ sbit + ".csv"

    if bit == 1:
        key = np.loadtxt(rand).reshape(1,size)
    else:
        key = np.loadtxt(rand).reshape(bit,size)


    key = tf.convert_to_tensor(key, np.float32)

    x_w = tf.matmul(key, wm_w)
    x_weights = tf.sigmoid(x_w)

    tf.print(x_weights, summarize=-1)

    if reg == "k":
        np.savetxt("Lkernel_weights.txt", x_weights)
    elif reg == "r":
        np.savetxt("Lrecurrent_weights.txt", x_weights)



def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

# sin波にノイズを付与する
def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

f = toy_problem()

def make_dataset(low_data, n_prev=100):

    data, target = [], []
    maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target


def acc(y_true, y_pred):
    return K.mean(y_true)

#g -> 学習データ，h -> 学習ラベル
g, h = make_dataset(f)

# モデル構築

# 1つの学習データのStep数(今回は25)
length_of_sequence = g.shape[1]
in_out_neurons = 1
n_hidden = 300

model = Sequential()

# 電子透かしの埋め込み先はここで指定する.
# kernel部:kernel_regularizer, recurrent部:recurrent_regularizer
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False, recurrent_regularizer=custom_l2_regularizer))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.01)
model.compile(loss="mean_squared_error", optimizer=optimizer)
model.summary()
history = model.fit(g, h,
          batch_size=100,
          epochs=100,
          validation_split=0.1,
          )


layer = model.layers[0]     #summaryよりInput->[0], Dense->[1]なのでmodel.layers[1]

# plt.ylim(0., 20.0)
plot_history_loss(history)
plt.show()
plt.close()


# 予測
predicted = model.predict(g)

plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.legend()
plt.show()

future_test = g[0]

# 1つの学習データの時間の長さ -> 25
time_length = future_test.shape[0]
# 未来の予測データを保存していく変数
future_result = np.empty((1))

# 未来予想
for step2 in range(400):

    test_data = np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)

    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)


# sin波をプロット
plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.plot(range(0+len(f), len(future_result)+len(f)), future_result, color="g", label="future_predict")
plt.legend()
plt.show()

# 電子透かしの検出
k_w = model.weights[0]
r_w = model.weights[1]

watermark_check(k_w, "k")
watermark_check(r_w, "r")


model.save("sin_noise.h5")


