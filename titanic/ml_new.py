import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .mypreprocessing import data_clean
# from mypreprocessing import data_clean
import warnings

warnings.filterwarnings("ignore")


class MachineLearn_new:
    """'Pclass', 'SibSp', 'Parch', 'Embarked', 'Initial',
       'Age_new', 'Sex_male', 'Sex_female', 'Fare_new', 'Family_size',
       'Alone'"""

    def __init__(self, sex, initial, age, sibsp, parch, fare, embarked, pclass):
        self.pclass = int(pclass)
        self.sibsp = int(sibsp)
        self.parch = int(parch)
        self.embarked = int(embarked)
        self.initiall = int(initial)
        age = float(age)
        if age < 22:
            self.age = 0
        elif age < 30:
            self.age = 1
        elif age < 36:
            self.age = 2
        else:
            self.age = 3

        if int(sex):
            self.male = 0
            self.female = 1
        else:
            self.male = 1
            self.female = 0

        fare = float(fare)
        if fare < 7.910400:
            self.fare = 0
        elif fare < 14.454200:
            self.fare = 1
        elif fare < 31.000000:
            self.fare = 2
        elif fare < 271.5:
            self.fare = 3
        else:
            self.fare = 4

        self.familysize = self.sibsp + self.parch

        if self.familysize > 0:
            self.alone = 0
        else:
            self.alone = 1

        """'Pclass', 'SibSp', 'Parch', 'Embarked', 'Initial',
               'Age_new', 'Sex_male', 'Sex_female', 'Fare_new', 'Family_size',
               'Alone'"""
        self.pred = [[self.pclass, self.sibsp, self.parch,
                      self.embarked, self.initiall, self.age, self.male, self.female, self.fare, self.familysize,
                      self.alone]]

        # 1、构造数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        FILE_PATH = PATH + "data" + os.sep + "titanic.csv"  # 处理前的数据
        SAVA_PATH = PATH + "data" + os.sep + "ml_new.csv"  # 处理后的数据
        if not os.path.exists(SAVA_PATH):
            data_clean(FILE_PATH, SAVA_PATH)  # 第一次的时候进行数据处理

        # 读取处理后的数据
        data = pd.read_csv(SAVA_PATH)
        x = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        self.x = x
        self.y = y
        # print(x.head())
        # print(y.head())
        # 训练集和测试集的切分
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=8)

    def mydel(self, file_path):
        try:
            os.remove(file_path)
        except:
            pass

    def KNN(self):
        """"
        年龄，小孩，票价，性别，独身
        """
        # 2、构造模型
        # model = KNeighborsClassifier()
        # #todo：超参数调整
        # hyper = {"n_neighbors":list(range(2,50))}
        # model = GridSearchCV(estimator=model,param_grid=hyper,verbose=True)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #n_neighbors=14

        model = KNeighborsClassifier(n_neighbors=14)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"KNN模型正确率：{score}"
        return str1, str2

    def LogicRegression(self):
        model = LogisticRegression()
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"Log模型正确率：{score}"
        return str1, str2

    def DecisionTree(self):
        # 2、构造模型
        # model = DecisionTreeClassifier()
        # #todo：超参数调整
        # hyper = {"criterion":["gini","entropy"]}
        # model = GridSearchCV(estimator=model,param_grid=hyper,verbose=True)
        # model.fit(self.x_train,self.y_train)
        # print(model.best_estimator_) #criterion='gini'

        model = DecisionTreeClassifier(criterion='gini')
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"DEC模型正确率：{score}"
        return str1, str2

    def RandomForest(self):
        # # 2、构造模型
        # model = ensemble.RandomForestClassifier()
        # #todo：超参数调整
        # hyper = {"criterion":["gini","entropy"],"n_estimators":list(range(10,100,10))}
        # scoring_func = metrics.make_scorer(metrics.f1_score)
        # #交叉验证
        # kfold = KFold(n_splits=5)
        # model = GridSearchCV(estimator=model,param_grid=hyper,scoring=scoring_func,cv=kfold,verbose=True)
        # model.fit(self.x,self.y)
        # print(model.best_estimator_) #criterion='gini'
        # print(model.best_score_)

        model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=90)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"RF模型正确率：{score}"
        return str1, str2

    def SVM(self):
        # 2、构造模型
        # model = svm.SVC()
        # #todo：超参数调整
        # hyper = {"gamma":np.arange(1,11)/10,"C":np.arange(1,11)/10,"kernel":["linear","poly","rbf"]}
        # scoring_func = metrics.make_scorer(metrics.f1_score)
        # #交叉验证
        # kfold = KFold(n_splits=5)
        # model = GridSearchCV(estimator=model,param_grid=hyper,scoring=scoring_func,cv=kfold,verbose=True)
        # model.fit(self.x,self.y)
        # print(model.best_estimator_) #gamma=0.6, kernel='rbf' C=0.3
        # print(model.best_score_)
        #

        model = svm.SVC(gamma=0.6, kernel='rbf', C=0.3)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"SVM模型正确率：{score}"
        return str1, str2

    def Bagging(self):

        # model = ensemble.BaggingClassifier(svm.SVC(gamma=0.6, kernel='rbf',C=0.3))
        model = ensemble.BaggingClassifier(KNeighborsClassifier(n_neighbors=14), n_estimators=200)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"Bag模型正确率：{score}"
        return str1, str2

    def Adaboost(self):

        model = ensemble.AdaBoostClassifier(n_estimators=200)
        # 3、模型训练
        model.fit(self.x_train, self.y_train)
        # 4、模型预测
        pred_y = model.predict(self.x_test)

        # 5、模型评估
        score = round(metrics.accuracy_score(self.y_test, pred_y), 2)
        # 6、模型应用
        pred = model.predict(self.pred)[0]
        if pred:
            pred = "-存活-"
        else:
            pred = "-死亡-"

        # 返回结果：str
        str1 = f"预测结果：{pred}"
        # 返回模型评估结果
        str2 = f"Ada模型正确率：{score}"
        return str1, str2


#
# m = MachineLearn_new(1, 2, 24, 1, 2, 200, 1, 1)
# print(m.KNN())
# print(m.LogicRegression())
# print(m.DecisionTree())
# print(m.RandomForest())
# print(m.SVM())
# print(m.Bagging())
# print(m.Adaboost())
