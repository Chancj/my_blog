import keras
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import random
import math


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=1, shuffle=True):
        self.BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.batch_size = batch_size
        # print(datas)

    def __len__(self):
        # print(math.ceil(len(self.datas) / float(self.batch_size)))
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_datas)
        # print(len(X), len(y))
        return X, y

    def data_generation(self, batch_datas):
        images = []
        labels = []
        for i, data in enumerate(batch_datas):
            img = image.load_img(data, target_size=(128, 128))
            x = image.img_to_array(img)
            x /= 255
            images.append(x)

            right = data.rfind("\\", 0)
            left = data.rfind("\\", 0, right) + 1
            # data_list = data.split(os.path.sep)
            calss_name = data[left:right]
            if calss_name == "dog":
                labels.append([0, 1])
            else:
                labels.append([1, 0])
        return np.array(images), np.array(labels)

    # # 对单张图像变形
    # def image_tran(self, img, batch_size=1, seed=1):
    #     datagen = ImageDataGenerator(
    #         rescale=1. / 255,
    #         rotation_range=20,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         shear_range=0.2,
    #         zoom_range=0.5,
    #         horizontal_flip=True,
    #         fill_mode="nearest",
    #     )
    #     for batch in datagen.flow(img, batch_size=batch_size, seed=seed):
    #         return batch

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.indexes)

