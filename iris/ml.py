from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cluster
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')


class MachineLearn(object):
    """iris数据集介绍：
    150个数据，分为三类，每类都是50个数据，每个数据包含：
    四个属性：花萼长度，花萼宽度，花瓣长度，花瓣宽度
    三个类别:山鸢尾，杂色鸢尾，维吉尼亚鸢尾花
    sepal width:花萼度度，单位cm
    sepal length:花萼长度，单位cm
    petal width:花瓣度度，单位cm
    spetal length:花瓣长度，单位cm
    Iris-setosa：山鸢尾
    Iris-versicolor：杂色鸢尾
    Iris-virginica：维吉尼亚鸢尾花"""

    def __init__(self):
        # 1、构造数据
        self.PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        FILE_PATH = self.PATH + "data" + os.sep + "iris.data"
        # print(FILE_PATH)
        data = pd.read_csv(FILE_PATH, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
        # todo:花瓣宽度和长度关系进行线性拟合
        reg_x = np.array(data.loc[:, "petal_width"]).reshape(-1, 1)
        reg_y = data.loc[:, "petal_length"]
        # 训练集和测试集的切分
        self.reg_x_train, self.reg_x_test, self.reg_y_train, \
        self.reg_y_test = train_test_split(reg_x, reg_y, test_size=0.2, random_state=8)

        # todo：分类数据集的构建
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # print(x.head())
        # print(y.head())
        # 训练集和测试集集的分离
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=8)

        # todo:聚类数据集
        self.km_x = x
        self.km_y = y

    def LineRegression(self, petal_width):
        """线性回归模型"""
        # 2、构造模型
        model = LinearRegression()
        # 3、模型训练
        model.fit(self.reg_x_train, self.reg_y_train)
        # 4、模型预测
        pred_y = model.predict(self.reg_x_test)
        # 5、模型评估
        score = round(metrics.r2_score(self.reg_y_test, pred_y), 2)
        # 6、模型应用
        pred = round(model.predict(np.array(petal_width).reshape(-1, 1))[0], 2)

        # 返回结果：预测结果
        str = f"预测结果：{pred}    线性模型r方得分：{score}"
        return str

    def PolyRegresson(self, petal_width):
        """多项式回归模型"""
        # 2、构造模型
        model = pipeline.make_pipeline(PolynomialFeatures(5), LinearRegression())  # 最高次幂为5次的多项式回归
        # 3、模型训练
        model.fit(self.reg_x_train, self.reg_y_train)
        # 4、模型预测
        pred_y = model.predict(self.reg_x_test)
        # 5、模型评估
        score = round(metrics.r2_score(self.reg_y_test, pred_y), 2)
        # 6、模型应用
        pred = round(model.predict(np.array(petal_width).reshape(-1, 1))[0], 2)

        # 返回结果：str
        str = f"预测结果：{pred}    多项式模型r方得分：{score}"
        return str

    def KNN(self, pred=[[1, 1, 1, 1]]):
        # knn_model = KNeighborsClassifier()
        # #todo:模型调参
        # hyper = {"n_neighbors":list(range(1,31))}
        # model = GridSearchCV(estimator=knn_model,param_grid=hyper)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #最佳nneighbors=3

        # 构造knn模型
        model = KNeighborsClassifier(n_neighbors=3)
        # 模型训练
        model.fit(self.x_train, self.y_train)
        # 模型预测
        pred_y = model.predict(self.x_test)
        # 模型评估
        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(pred)[0]
        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"KNN模型正确率：{score}"
        return str1, str2

    def LogsticRegression(self, pred=[[1, 1, 1, 1]]):

        # 构造LogsticRegression模型
        model = LogisticRegression()
        # 模型训练
        model.fit(self.x_train, self.y_train)
        # 模型预测
        pred_y = model.predict(self.x_test)
        # 模型评估
        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(pred)[0]
        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"Log模型正确率：{score}"
        return str1, str2

    def DecideTree(self, pred=[[1, 1, 1, 1]]):
        # dec_model = DecisionTreeClassifier()
        # #todo:模型调参
        # hyper = {"max_depth":list(range(1,5)),"criterion":["gini","entropy"]}
        # model = GridSearchCV(estimator=dec_model,param_grid=hyper)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #criterion='entropy', max_depth=4

        # 构造DecideTree模型
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        # 模型训练
        model.fit(self.x_train, self.y_train)
        # 模型预测
        pred_y = model.predict(self.x_test)
        # 模型评估
        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(pred)[0]
        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"Dec模型正确率：{score}"
        return str1, str2

    def RandomForest(self, pred=[[1, 1, 1, 1]]):
        # rf_model = RandomForestClassifier(random_state=8)
        # #todo:模型调参
        # hyper = {"n_estimators":list(range(10,100,10)),"criterion":["gini","entropy"]}
        # model = GridSearchCV(estimator=rf_model,param_grid=hyper)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #criterion='gini', n_estimators=10
        # print(model.best_score_)

        # 构造RandomForestClassifier模型
        model = RandomForestClassifier(criterion='gini', n_estimators=10)
        # 模型训练
        model.fit(self.x_train, self.y_train)
        # 模型预测
        pred_y = model.predict(self.x_test)
        # 模型评估
        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(pred)[0]
        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"RF模型正确率：{score}"
        return str1, str2

    def SVM(self, pred=[[1, 1, 1, 1]]):
        # svm_model = svm.SVC()
        # #todo:模型调参 C=1.0, kernel='rbf',
        # hyper = {"kernel":["linear","poly","rbf"]}
        # model = GridSearchCV(estimator=svm_model,param_grid=hyper)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #kernel='linear'
        # print(model.best_score_)

        # 构造RandomForestClassifier模型
        model = svm.SVC(kernel='linear')
        # 模型训练
        model.fit(self.x_train, self.y_train)
        # 模型预测
        pred_y = model.predict(self.x_test)
        # 模型评估
        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(pred)[0]
        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"svm模型正确率：{score}"
        return str1, str2

    def Cluster(self, pred=[[1, 1, 1, 1]]):
        """无监督学习"""
        # 构造无监督学习模型
        # model = cluster.KMeans(n_clusters=3)  #随机中心点，可以归类，但输出结果可能每次都不同
        model = cluster.MeanShift()  # 自我归类，最佳匹配
        # 模型训练
        model.fit(self.km_x)
        # 模型评估:轮廓系数
        score = round(metrics.silhouette_score(self.km_x, model.labels_), 2)
        # 模型应用
        pred = model.predict(pred)[0]
        pred = str(pred)  # 强转一下格式
        if pred == '0':
            pred = "Iris-virginica"
        elif pred == '1':
            pred = "Iris-setosa"
        else:
            pred = "Iris-versicolor"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"聚类轮廓系数：{score}"
        return str1, str2

#
# m = MachineLearn()
# # print(m.LineRegression(3))
# # print(m.PolyRegresson(2))
# print(m.KNN([[7,3,5,2]]))
# print(m.LogsticRegression([[7,3,5,2]]))
# print(m.DecideTree([[7,3,5,2]]))
# print(m.RandomForest([[7,3,5,2]]))
# print(m.SVM([[7,3,5,2]]))
# print(m.Cluster([[7,3,5,2]]))
