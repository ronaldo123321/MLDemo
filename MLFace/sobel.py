from keras.models import Sequential
from keras.layers import Convolution2D
from keras.callbacks import Callback
from PIL import Image
import numpy as np
from scipy.ndimage.filters import convolve

#用lina图训练sobel算子

class LossHistory(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.losses = []
    def on_train_begin(self, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

# lena=np.array(Image.open(r"C:\Users\anytec-gb\Desktop\lina.jpg").convert("L"))
lena=np.array(Image.open("lina.jpg").convert("L"))
lena_sobel=np.zeros(lena.shape)

#sobel算子
sobel = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

#计算卷积  用 sobel 算子滤波。
convolve(input=lena,output=lena_sobel,weights=sobel,mode="constant",cval=1.0)

#将像素值调整到[0,255]区间并保存sobel算子滤波后的lena图
lena_tmp=np.uint8((lena_sobel - lena_sobel.min()) * 255 / (lena_sobel.max() - lena_sobel.min()))
Image.fromarray(lena_tmp).save(lena_sobel.jpg)

# 将原始 lena 图和 sobel 滤波 lena 图转换成 (1, 1, width, height) 尺寸。第一个 1 表示训练集只有一个样本。第二个 1 表示样本只有一个 channel 。
X = lena.reshape((1, 1) + lena.shape)
Y = lena_sobel.reshape((1, 1) + lena_sobel.shape)

# 建一个神经网络模型。
model = Sequential()

# 只添加一个卷积层。卷积层只有一个滤波器。滤波器尺寸 3x3 。输入维度顺序是 "th" 表示 (channel, width, height) 。输入尺寸是 (channel, width, height) 。不要偏执置。
model.add(
    Convolution2D(nb_filter=1, nb_row=3, nb_col=3, dim_ordering="th", input_shape=X.shape[1:], border_mode="same",
                  bias=False, init="uniform"))

# 代价函数取 mse 。优化算法取 rmsprop 。
model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

history = LossHistory()

# 训练 10 轮，每轮保存一下当前网络输出图像。
for i in np.arange(0, 10):
    lena_tmp = model.predict(X).reshape(lena.shape)
    lena_tmp = np.uint8((lena_tmp - lena_tmp.min()) * 255 / (lena_tmp.max() - lena_tmp.min()))
    Image.fromarray(lena_tmp).save("lena_sobel_stage_{:d}.jpg".format(i))
    print("lena_sobel_stage_{:d}.png saved".format(i))

    model.fit(X, Y, batch_size=1, nb_epoch=200, verbose=1, callbacks=[history])
    print("Epoch {:d}".format(i + 1))

lena_tmp = model.predict(X).reshape(lena.shape)
lena_tmp = np.uint8((lena_tmp - lena_tmp.min()) * 255 / (lena_tmp.max() - lena_tmp.min()))
Image.fromarray(lena_tmp).save("lena_sobel_stage_final.jpg")

