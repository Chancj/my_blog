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
import warnings

warnings.filterwarnings("ignore")


class MachineLearn:
    def __init__(self, sex, age, fare, alone):
        # 1、构造数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        FILE_PATH = PATH + "data" + os.sep + "titanic.csv"  # 处理前的数据
        SAVA_PATH = PATH + "data" + os.sep + "ml.csv"  # 处理后的数据
        if not os.path.exists(SAVA_PATH):
            data_train = pd.read_csv(FILE_PATH)
            # 处理年龄空值
            # 获取称呼，根据称呼归类，填充平均值
            data_train["Initial"] = 0  # 新增一列，获取称呼
            data_train["Initial"] = data_train["Name"].str.extract('(\w+)\.')
            # print(data_train["Initial"].unique())
            # ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms'
            # 'Major' 'Lady''Sir' 'Mlle' 'Col' 'Capt' 'Countess' 'Jonkheer']
            data_train["Initial"].replace(['Ms', 'Mme', 'Mlle'], ['Miss', 'Miss', 'Miss'], inplace=True)
            data_train["Initial"].replace(['Lady', 'Countess'], ['Mrs', 'Mrs'], inplace=True)
            data_train["Initial"].replace(['Jonkheer', 'Col', 'Rev'], ["other", "other", "other"], inplace=True)
            data_train["Initial"].replace(['Major', 'Capt', 'Sir', 'Don', 'Dr'], ['Mr', 'Mr', 'Mr', 'Mr', 'Mr'],
                                          inplace=True)
            # print(data_train.groupby("Initial")["Age"].mean())
            # Master 4.574167  Miss 21.860000   Mr 32.739609  Mrs 35.981818    other 45.888889
            # todo:填充年龄空值
            data_train["Age_new"] = data_train["Age"]  # 新生成一列，复制原来的年龄
            data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Master"), "Age_new"] = 5
            data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Miss"), "Age_new"] = 22
            data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Mr"), "Age_new"] = 33
            data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Mrs"), "Age_new"] = 36
            data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "other"), "Age_new"] = 46

            # todo:是否是小孩:年龄小于10岁
            data_train["Child_new"] = 0
            data_train.loc[data_train["Age_new"] <= 10, "Child_new"] = 1

            # print(data_train.head())
            # print(data_train["Age_new"].describe())
            # min  0.420000
            # 25 % 22.000000
            # 50 % 30.000000
            # 75 % 36.000000
            # max  80.000000

            # todo:年龄离散化
            data_train.loc[(data_train["Age_new"] >= 0.42) & (data_train["Age_new"] < 22), "Age_new"] = 0
            data_train.loc[(data_train["Age_new"] >= 22) & (data_train["Age_new"] < 30), "Age_new"] = 1
            data_train.loc[(data_train["Age_new"] >= 30) & (data_train["Age_new"] < 36), "Age_new"] = 2
            data_train.loc[(data_train["Age_new"] >= 36) & (data_train["Age_new"] <= 80), "Age_new"] = 3

            # todo:费用离散化
            # print(data_train["Fare"].describe())
            # min  0.000000
            # 25 % 7.910400
            # 50 % 14.454200
            # 75 % 31.000000
            # max  512.329200
            data_train["Fare_new"] = 0
            data_train.loc[(data_train["Fare"] >= 0) & (data_train["Fare"] < 7.910400), "Fare_new"] = 0
            data_train.loc[(data_train["Fare"] >= 7.910400) & (data_train["Fare"] < 14.454200), "Fare_new"] = 1
            data_train.loc[(data_train["Fare"] >= 14.454200) & (data_train["Fare"] < 31.000000), "Fare_new"] = 2
            data_train.loc[(data_train["Fare"] >= 31.000000) & (data_train["Fare"] < 512.329200), "Fare_new"] = 3

            # todo:性别编码化
            data_train["Sex_new"] = 0  # 0表示男性
            data_train.loc[data_train["Sex"] == "female", "Sex_new"] = 1

            # todo:是否是独身
            data_train["Alone_new"] = 1
            data_train.loc[(data_train["SibSp"] + data_train["Parch"]) > 0, "Alone_new"] = 0  # SibSp,Parch,

            # print(data_train.columns)
            # todo：构造训练集和测试集
            train_df = data_train.filter(regex="Survived|.*_new")
            # print(train_df.head())
            # 保存数据
            train_df.to_csv(SAVA_PATH, index=False)
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

        # todo:加工预测数据
        if sex == "male":
            self.sex = 0
        else:
            self.sex = 1
        # min  0.420000
        # 25 % 22.000000
        # 50 % 30.000000
        # 75 % 36.000000
        # max  80.000000
        age = float(age)
        if age < 10:
            self.child = 1
        else:
            self.child = 0

        if age < 22:
            self.age = 0
        elif age < 30:
            self.age = 1
        elif age < 36:
            self.age = 2
        else:
            self.age = 3

        # min  0.000000
        # 25 % 7.910400
        # 50 % 14.454200
        # 75 % 31.000000
        # max  512.329200
        fare = float(fare)
        if fare < 7.910400:
            self.fare = 0
        elif fare < 14.454200:
            self.fare = 1
        elif fare < 31.000000:
            self.fare = 2
        else:
            self.fare = 3
        self.alone = int(alone)

        self.pred = [[self.age, self.child, self.fare, self.sex, self.alone]]

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
m = MachineLearn('male', "22", "15", 1)  # sex,age,fare,alone
# print(m.KNN())
# print(m.LogicRegression())
# print(m.DecisionTree())
# print(m.RandomForest())
# print(m.SVM())
# print(m.Bagging())
# print(m.Adaboost())
