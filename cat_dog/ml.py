import os
import numpy as np
import shutil
import random
# todo;构造卷积神经网络
from keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten
from keras.models import load_model, Sequential
from keras.preprocessing import image
# from data_gen import DataGenerator
from .data_gen import DataGenerator


class CatDog():
    def __init__(self, file_path):
        self.file_path = file_path
        # 构造训练数据
        self.BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        self.original_data = os.path.join(self.BATH_PATH, "data", "original_data") + os.sep
        # print(self.train) #D:\Users\Administrator\Desktop\ml_project\ai_yueqian\catdog\data\train\
        # 猫狗数据分离
        # 构造目标训练地址
        self.target = os.path.join(self.BATH_PATH, "data", "target") + os.sep
        # print(self.target) #D:\Users\Administrator\Desktop\ml_project\ai_yueqian\catdog\data\target\
        # todo:构造模型保存路径
        self.model_path = os.path.join(self.BATH_PATH, "static", "model") + os.sep
        self.model_save = os.path.join(self.BATH_PATH, "static", "model", "mymodel.h5")
        self.model_save_weights = os.path.join(self.BATH_PATH, "static", "model", "myweights.h5")

    def ensure_dir(self, dir_path):
        """
        创建文件夹
        :param dir_path: 文件夹的路径
        :return:
        """
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError:
                print("文件夹已经存在，创建文件夹失败")

    def del_dir(self, dir_path):
        """
        删除文件夹
        :param dir_path: 文件夹的路径
        :return:
        """
        try:
            shutil.rmtree(dir_path)  # 删除当前文件夹所有目录
        except FileNotFoundError:
            print(f"{dir_path}路径不存在！")

    def init_cat_dog(self, fresh=False):
        """
        构造数据集
        :param fresh: 是否重新构造数据集
        :return:
        """
        if fresh:
            try:
                self.del_dir(self.target)
            except Exception:
                print(f"删除路径失败：{self.target}")

        if not os.path.exists(self.target):
            # 创建保存训练数据的路径
            self.ensure_dir(os.path.join(self.target, "train", "cat") + os.sep)
            self.ensure_dir(os.path.join(self.target, "train", "dog") + os.sep)
            self.ensure_dir(os.path.join(self.target, "test", "cat") + os.sep)
            self.ensure_dir(os.path.join(self.target, "test", "dog") + os.sep)

            # todo:训练集和测试集的分离
            train_list = os.listdir(self.original_data)  # 路径下的所有文件名称
            # print(train_list)
            dogs = [self.original_data + i for i in train_list if "dog" in i]
            # print(dogs)
            cats = [self.original_data + i for i in train_list if "cat" in i]
            # 复制到数据到训练路径中
            random.shuffle(dogs)
            random.shuffle(cats)
            cut_size = int(len(dogs) * 0.75)  # 75%训练
            # todo:构造训练数据
            for dog_path in dogs[:cut_size]:
                shutil.copyfile(dog_path,
                                os.path.join(self.target, "train", "dog") + os.sep + os.path.basename(dog_path))
                shutil.copyfile(dog_path,
                                os.path.join(self.target, "test", "dog") + os.sep + os.path.basename(dog_path))
            for cat_path in cats[:cut_size]:
                shutil.copyfile(cat_path,
                                os.path.join(self.target, "train", "cat") + os.sep + os.path.basename(cat_path))
                shutil.copyfile(cat_path,
                                os.path.join(self.target, "test", "cat") + os.sep + os.path.basename(cat_path))
        else:
            print("训练集和数据集已经就绪，不需要重复加载！")

    def init_data(self, datatype="train"):
        """
        读取数据
        :param datatype: 读取数据的文件夹[‘train’,'test']
        :return:所有训练数据的路径
        """
        datas = []
        data_path = self.target + datatype + os.sep
        for file in os.listdir(data_path):
            # print(file) #["cat","dog"]
            file_path = os.path.join(data_path, file)
            # print(file_path)
            if os.path.isdir(file_path):
                for subfile in os.listdir(file_path):
                    datas.append(os.path.join(file_path, subfile))
        return datas

    def init_model(self):
        """
        构造卷积神经网络模型
        :return:
        """
        # 统一图像尺寸
        img_width = 128
        img_height = 128
        input_shape = (img_width, img_height, 3)
        model = Sequential([
            Convolution2D(32, (3, 3), input_shape=input_shape, strides=(1, 1), activation="relu"),  # 卷积层
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool1"),  # 池化层
            Convolution2D(64, (3, 3), activation="relu"),  # 卷积层
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool2"),  # 池化层
            Flatten(),  # 扁平化
            Dense(64, activation="relu"),
            Dropout(0.5),  # 随机失活
            Dense(2, activation="sigmoid")
        ])
        # 编译模型
        # 动量梯度下降算法，交叉熵误差函数，正确率做模型评估指标
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model

    def save_my_model(self):
        """
        保存模型
        :return:
        """
        self.ensure_dir(self.model_path)
        # json_string = self.model.to_json()
        self.model.save(self.model_save)  # 保存模型
        self.model.save_weights(self.model_save_weights)  # b保存权重

    def load_my_model(self):
        model = load_model(self.model_save)  # 加载模型
        model.load_weights(self.model_save_weights)  # 加载权重
        return model

    def model_trian(self, refresh=False):
        """
        模型训练
        :return:
        """
        if refresh:
            try:
                self.del_dir(self.model_path)
            except Exception:
                print(f"删除路径失败：{self.model_path}")
        if not os.path.exists(self.model_save) or not os.path.exists(self.model_save_weights):

            # 1、构造数据
            self.init_cat_dog()
            train_datas = self.init_data()
            train_generator = DataGenerator(train_datas, batch_size=32, shuffle=True)
            # 2、构造模型
            self.init_model()
            # 3、模型训练
            self.model.fit_generator(train_generator, epochs=30, max_queue_size=10, workers=1, verbose=1)
            # 保存模型
            self.save_my_model()
        else:
            import tensorflow as tf
            graph = tf.get_default_graph()  # 功能：获取当前默认计算图。
            with graph.as_default():

                self.model = self.load_my_model()

    def pred_cat_dog(self):
        img = image.load_img(self.file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)

        y = self.model.predict(x)

        pred_index = np.argmax(y)

        if pred_index == 1:
            return "识别结果：-<狗狗>-"
        else:
            return "识别结果：-<喵喵>-"

#
# c = CatDog('./2.jpg')
# c.model_trian()
# print(c.pred_cat_dog())
