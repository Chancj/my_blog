from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.svm import SVC
import sklearn.utils as su
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
import neurolab
# 创建深度学习模型
from keras.models import Sequential, load_model
from keras.layers import Dense
# 对标签one-hot编码
from keras.utils import np_utils
from keras.layers.core import Dropout  # 随机失活
from keras.layers.core import Activation
import tensorflow as tf
import random

import warnings

warnings.filterwarnings('ignore')


class DeepLearn():
    def __init__(self, file_path, pca_k):
        # 1.构造数据
        # todo:训练集
        BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        PATH = BASE_PATH + "att_faces" + os.sep
        IMG_H = 112
        IMG_W = 92
        self.img_size = IMG_H * IMG_W
        self.biach_size = 28
        self.nb_epochs = 1000
        self.pca_k = int(pca_k)
        x = []
        y = []
        # 将照片导入numpy数组，然后将他们的像素矩阵替换为向量
        for dir_path, dir_names, file_names in os.walk(PATH):
            for fn in file_names:
                # if fn.endswith("pgm"):
                if fn[-3:] == "pgm":
                    image_filename = os.path.join(dir_path, fn)
                    # 图像处理
                    img = Image.open(image_filename).convert("L")  # 灰化
                    im_arr = np.array(img)
                    # print(im_arr.shape)  #112*92=10304
                    x_d = im_arr.reshape(10304).astype("float32") / 255  # 量化
                    x.append(x_d)
                    y.append(dir_path)

        x = np.array(x)
        # 训练集和测试集的分离
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=7)
        # todo:pca降维
        pca = PCA(n_components=pca_k)
        self.x_train = pca.fit_transform(self.x_train)
        self.x_test = pca.transform(self.x_test)
        # todo:还原
        # self.x_train = pca.inverse_transform(self.x_train)
        # self.x_test = pca.inverse_transform(self.x_test)

        # 处理标签集
        yy_train = []
        for i in range(len(self.y_train)):
            yy = int(str(self.y_train[i].split('\\')[-1])[1:])
            # print(yy)
            yy_train.append(yy)
        self.y_train = np.array(yy_train).reshape(-1, 1)
        yy_test = []
        for i in range(len(self.y_test)):
            yy = int(str(self.y_test[i].split('\\')[-1])[1:])
            yy_test.append(yy)
        self.y_test = np.array(yy_test).reshape(-1, 1)
        # print(self.y_train[:5])
        # print(self.y_test[:5])

        # 神经网络标签;独热编码
        self.ohe = OneHotEncoder(sparse=False, dtype=int)
        self.nn_y_train = self.ohe.fit_transform(self.y_train)
        self.nn_y_test = self.ohe.transform(self.y_test)

        # todo：构造测试集的数据
        im = Image.open(file_path).convert("L")
        im_arr = np.array(im)
        # pre_face = scale(im_arr.reshape(10304).astype("float32"))
        pre_face = im_arr.reshape(-1).astype("float32") / 255
        x_pred = [pre_face]
        x_pred = np.array(x_pred)
        self.pred = pca.transform(x_pred)  # 降维处理
        # self.pred = pca.inverse_transform(self.pred) #还原

    # 6、模型保存及应用
    def save_model(self, model, directory):
        model.save(directory)

    def load_model(self, directory):
        model = load_model(directory)
        return model

    def Keras_DNN(self, refresh=False):

        BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        if self.pca_k == 150:
            MODEL_PATH = BATH_PATH + "static" + os.sep + "model" + os.sep + "dnn150.h5"
        else:
            MODEL_PATH = BATH_PATH + "static" + os.sep + "model" + os.sep + "dnn200.h5"
        if refresh:
            os.remove(MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            model = Sequential([
                Dense(512, input_shape=(self.pca_k,), activation="softmax"),
                Activation("relu"),
                Dropout(0.2),
                Dense(512, input_shape=(512,), activation="softmax"),
                Activation("relu"),
                Dropout(0.2),  # 随机失活
                Dense(40, input_shape=(512,), activation="softmax"),
            ])
            # 编译模型
            model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
            # 3、模型训练
            model.fit(self.x_train, self.nn_y_train, batch_size=self.biach_size, epochs=self.nb_epochs, verbose=1,
                      validation_data=(self.x_test, self.nn_y_test))
            self.save_model(model, MODEL_PATH)  # 保存模型
        else:
            model = self.load_model(MODEL_PATH)

        # 4、模型预测
        pred = model.predict(self.pred)
        pred = pred.argmax() + 1
        pred = 'DNN预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        score = model.evaluate(self.x_test, self.nn_y_test, verbose=0)
        metr = "正确率：" + str(round(score[1], 2))
        return pred, metr

# # #
# c = DeepLearn('./10.pgm',200)
# # #
# print(c.Keras_DNN())
#
#
#
#
