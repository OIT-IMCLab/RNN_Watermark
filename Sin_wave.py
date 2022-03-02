import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
import tensorflow as tf
import keras.backend as K

# 埋め込みbit数の指定
bit = 1

# シグモイド関数の定義
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# lossのグラフ
def plot_history_loss(fit):
    # Plot the loss in the history
    plt.plot(fit.history['loss'],label="loss for training")
    plt.plot(fit.history['val_loss'],label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

# L2正則化
# 適用する場合はlamの値を"0."から変更
def l2_reg(w) :

    lam = 0.
    pe = lam * K.sum(K.square(w))

    return pe

def custom_l2_regularizer(w):

    print("w_shape:" + str(w.get_shape()[0]) + ", " + str(w.get_shape()[1]))
    # 透かし埋め込みの強度
    lam = 0.04
    sbit = str(bit)
    size = w.get_shape()[0] * w.get_shape()[1]
    print(size)

    # 重み係数のベクトル化
    wm_w = K.reshape(w, [-1, 1])

    # 重み係数群の個数で埋め込み先を判別
    if not size == 400 :
        rand = "K_test"+ sbit + ".csv"
    else:
        rand = "R_test"+ sbit +".csv"

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
    wm_w = K.reshape(w, [size, 1])

    if not size == 400:
        rand = "K_test"+ sbit + ".csv"
    else:
        rand = "R_test"+ sbit +".csv"

    if bit == 1:
        key = np.loadtxt(rand).reshape(1,size)
    else:
        key = np.loadtxt(rand).reshape(bit,size)

    key = tf.convert_to_tensor(key, np.float32)

    x_w = tf.matmul(key, wm_w)
    x_weights = tf.sigmoid(x_w)

    print(x_weights)

    if reg == "k":
        np.savetxt("kernel_weights.txt", x_weights)
    elif reg == "r":
        np.savetxt("recurrent_weights.txt", x_weights)



def sin2p(x, t=100):
    return np.sin(2.0 * np.pi * x / t)  # sin(2πx/t) t = 周期


def sindata(t=100, cycle=2):
    x = np.arange(0, cycle * t)  # 0 から cycle * t 未満の数
    return sin2p(x)


def noisy(Y, noise_range=(-0.2, 0.2)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise, noise



# データ準備
# np.random.seed(0)
rdata = noisy(sindata(100, 2), (-0.2, 0.2))
inputlen = 25

rawdata = rdata[0]
noisedata = rdata[1]

rawdata = sindata(100,2)

input = []
target = []
for i in range(0, len(rawdata) - inputlen):
    input.append(rawdata[i:i + inputlen])
    target.append(rawdata[i + inputlen])

X = np.array(input).reshape(len(input), inputlen, 1)
Y = np.array(target).reshape(len(input), 1)
x, val_x, y, val_y = train_test_split(X, Y, test_size=int(len(X) * 0.2), shuffle=False)

n_in = 1
n_hidden = 20
n_out = 1
epochs = 15
batch_size = 10

model = Sequential()

# 電子透かしの埋め込み先はここで指定する.
# kernel部:kernel_regularizer, recurrent部:recurrent_regularizer
model.add(SimpleRNN(n_hidden, input_shape=(inputlen, n_in), kernel_initializer='random_normal', kernel_regularizer=custom_l2_regularizer))

model.add(Dense(n_out, kernel_initializer='random_normal'))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
model.summary()

history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))
model.save_weights('nashi_weight.h5')

# 予測
in_ = x[:1]  # x の先頭 (1,20,1) 配列
predicted = [None for _ in range(inputlen)]
for _ in range(len(rawdata) - inputlen):
    out_ = model.predict(in_)  # 予測した値 out_ は (1,1) 配列
    in_ = np.concatenate((in_.reshape(inputlen, n_in)[1:], out_), axis=0).reshape(1, inputlen, n_in)
    predicted.append(out_.reshape(-1))

plt.title('Predict sin wave')
plt.plot(rawdata, label="original")
plt.plot(predicted, label="predicted")
plt.plot(x[0], label="input")
plt.legend()
plt.show()

# ----------------------------------------------
# Some plots
# ----------------------------------------------
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


plot_history_loss(history)
plt.show()
plt.close()


# 以下、電子透かしの検出
k_w = model.weights[0]
r_w = model.weights[1]

watermark_check(k_w, "k")
watermark_check(r_w, "r")

model.save('SimpleRNN.h5')