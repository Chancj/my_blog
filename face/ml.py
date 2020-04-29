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

import warnings

warnings.filterwarnings('ignore')


class MachineLearn():
    def __init__(self, file_path, pca_k):
        # 1.构造数据
        # todo:训练集
        BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        PATH = BASE_PATH + "att_faces" + os.sep
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
                    # x_d = im_arr.reshape(10304).astype("float32")/255 #量化
                    x_d = scale(im_arr.reshape(10304).astype("float32"))
                    x.append(x_d)
                    y.append(dir_path)

        # print(y)
        x = np.array(x)
        # 训练集和测试集的分离
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=7)
        # todo:pca降维
        pca = PCA(n_components=pca_k)
        self.x_train = pca.fit_transform(self.x_train)
        self.x_test = pca.transform(self.x_test)
        # print(self.x_train.shape)
        # print(self.x_test.shape)
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
        pre_face = scale(im_arr.reshape(-1).astype("float32"))
        x_pred = [pre_face]
        x_pred = np.array(x_pred)
        self.pred = pca.transform(x_pred)  # 降维处理

    def KNN(self):
        # 2、构造模型
        model = KNeighborsClassifier(n_neighbors=1)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'KNN预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr
        # 6、模型保存及应用

    def LogsticRegressor(self):
        # 2、构造模型
        # l-bfgs:共轭梯度法，还有其他三种算法：liblinear,newton-cg,sag
        # 分类选择："ovr","multinomial",todo:multinomal一般用在多元逻辑回归上
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'Log预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr

    def DecisionTree(self):
        # es = DecisionTreeClassifier()
        # para = {"criterion":["gini","entropy"],"max_depth":range(1,50)}
        # scoring_func = metrics.make_scorer(metrics.accuracy_score)
        # # kfold = KFold(n_splits=5)
        # # grid = GridSearchCV(estimator=es,param_grid=para,scoring=scoring_func,cv=kfold)
        # grid = GridSearchCV(estimator=es,param_grid=para,scoring=scoring_func)
        # grid.fit(self.x_train,self.y_train)
        # acc = grid.best_score_
        # print(f"最优正确率为：{acc}")
        # reg = grid.best_estimator_
        # print(f"最优参数为：\n",reg)

        model = DecisionTreeClassifier(criterion='entropy', max_depth=40)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'Dec预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr

    def RandomForest(self):
        # es = ensemble.RandomForestClassifier(max_depth=30,criterion='entropy')
        # para = {"n_estimators":[500,600,700,800]}
        # scoring_func = metrics.make_scorer(metrics.accuracy_score)
        # grid = GridSearchCV(estimator=es,param_grid=para,scoring=scoring_func)
        # grid.fit(self.x_train,self.y_train)
        # acc = grid.best_score_
        # print(f"最优正确率为：{acc}")
        # reg = grid.best_estimator_
        # print(f"最优参数为：\n",reg)

        model = ensemble.RandomForestClassifier(max_depth=30, criterion='entropy', n_estimators=20, random_state=7)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'RF预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr

    def SVM(self):
        # # 构造最优模型
        # gamma = list(np.arange(0.01,0.1,0.01))
        # C = list(np.arange(0.1,1,0.1))
        # para = {"gamma":gamma,"C":C}
        # es = SVC(kernel="linear")
        # scoring_func = metrics.make_scorer(metrics.accuracy_score)
        # grid = GridSearchCV(estimator=es,param_grid=para,scoring=scoring_func)
        # grid.fit(self.x_train,self.y_train)
        # acc = grid.best_score_
        # print(f"最优正确率为：{acc}")
        # reg = grid.best_estimator_
        # print(f"最优参数为：\n",reg)

        model = SVC(kernel="linear", C=0.1, gamma=0.01)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'SVM预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr

    def Bagging(self):
        model = ensemble.BaggingClassifier(KNeighborsClassifier(n_neighbors=1), n_estimators=20)
        # model = ensemble.BaggingClassifier(SVC(kernel="linear",C=0.1,gamma=0.01),n_estimators=700)
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred = model.predict(self.pred)[0]
        pred = 'Bag预测：' + 's' + str(pred)
        # 5、模型评估
        pred_test_y = model.predict(self.x_test)
        metr = round(metrics.accuracy_score(self.y_test, pred_test_y), 2)
        metr = "正确率：" + str(metr)
        return pred, metr

# c = MachineLearn('./7.pgm',150)
# print(c.KNN())
# print(c.LogsticRegressor())
# print(c.DecisionTree())
# print(c.RandomForest())
# print(c.SVM())
# print(c.Bagging())
# print(c.DNN_python())
