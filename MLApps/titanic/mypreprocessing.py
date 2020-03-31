import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 设置行不限制数量
pd.set_option('display.max_rows', None)
# 2. 设置列不限制数量
pd.set_option('display.max_columns', None)
plt.style.use("fivethirtyeight")


def data_clean(file_path, save_path):
    #################(-)数据读取################
    # todo/1、加载数据
    data = pd.read_csv(file_path)
    # todo:2/年龄的离散化
    # 年龄空值的处理
    data["Initial"] = 0  # 新建一列特征保存Name中的称呼
    data["Initial"] = data["Name"].str.extract('(\w+)\.')
    # 称呼归类
    data['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)
    # 求出归类以后的年龄平均值
    # print(set(data["Initial"])) #{'Master', 'Mr', 'Other', 'Mrs', 'Miss'}
    # print(data.groupby("Initial")["Age"].mean())
    Master_age_mean = 5
    Miss_age_mean = 22
    Mr_age_mean = 33
    Mrs_age_mean = 36
    Other_age_mean = 46
    # 空值使用均值填充
    data.loc[(data["Age"].isnull()) & (data["Initial"] == "Master"), "Age"] = Master_age_mean
    data.loc[(data["Age"].isnull()) & (data["Initial"] == "Mr"), "Age"] = Mr_age_mean
    data.loc[(data["Age"].isnull()) & (data["Initial"] == "Other"), "Age"] = Other_age_mean
    data.loc[(data["Age"].isnull()) & (data["Initial"] == "Mrs"), "Age"] = Mrs_age_mean
    data.loc[(data["Age"].isnull()) & (data["Initial"] == "Miss"), "Age"] = Miss_age_mean
    # todo:年龄离散化
    data["Age_new"] = 0  # 新生成一列特征进行年龄离散化

    data.loc[(data["Age_new"] >= 0.42) & (data["Age_new"] < 22), "Age_new"] = 0
    data.loc[(data["Age_new"] >= 22) & (data["Age_new"] < 30), "Age_new"] = 1
    data.loc[(data["Age_new"] >= 30) & (data["Age_new"] < 36), "Age_new"] = 2
    data.loc[(data["Age_new"] >= 36) & (data["Age_new"] <= 80), "Age_new"] = 3

    # 通过五数概括来分类
    # print(data["Age"].describe())
    # min        0.420000
    # 25%       22.000000
    # 50%       30.000000
    # 75%       36.000000
    # max       80.000000
    # data.loc[data["Age"]<=22,"Age_new"] = 0
    # data.loc[(data["Age"]>22)&(data["Age"]<=30),"Age_new"] = 1
    # data.loc[(data["Age"]>30)&(data["Age"]<=36),"Age_new"] = 2
    # data.loc[data["Age"]>36,"Age_new"] = 3
    # print(data.loc[:,["Age","Age_new"]].head(20))

    # todo:登录港口
    # 空值处理
    data["Embarked"].fillna("S", inplace=True)
    # 登录港口编码化
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # todo:性别
    # 编码化
    # data.loc[data["Sex"]=="male","Sex"] = 0
    # data.loc[data["Sex"]=="female","Sex"] = 1
    # 独热编码化
    data["Sex_male"] = 0
    data["Sex_female"] = 0
    data.loc[data["Sex"] == "male", "Sex_male"] = 1
    data.loc[data["Sex"] == "female", "Sex_female"] = 1

    # todo:票价
    # print(data["Fare"].describe())
    # min        0.000000
    # 25%        7.910400
    # 50%       14.454200
    # 75%       31.000000
    # max      512.329200
    data["Fare_new"] = 0
    data.loc[data["Fare"] <= 7.910400, "Fare_new"] = 0
    data.loc[(data["Fare"] > 7.910400) & (data["Fare"] <= 14.454200), "Fare_new"] = 1
    data.loc[(data["Fare"] > 14.454200) & (data["Fare"] <= 31.000000), "Fare_new"] = 2
    data.loc[(data["Fare"] > 31) & (data["Fare"] <= 271.5), "Fare_new"] = 3
    data.loc[data["Fare"] > 271.5, "Fare_new"] = 4

    # todo:称呼
    data["Initial"].replace(
        ['Master', 'Miss', 'Mrs', 'Mr', 'Other'],
        [0, 1, 2, 3, 4], inplace=True)

    # todo；是否是独身
    # print(data.isnull().sum())
    data["Family_size"] = data["Parch"] + data["SibSp"]  # 新生成一列特征保存家庭总成员数
    data["Alone"] = 0  # 新生成一列保存是否独身
    data.loc[data["Family_size"] == 0, "Alone"] = 1  # 默认不独身，如果家庭成员数为0，则表示独身

    # print(data.head())
    # todo:建模准备
    # print(data.columns)
    # 'PassengerId',不能被归类，不保留
    #  'Survived', 标签数值，保留
    # 'Pclass',重要的特征，保留
    #  'Name', 不能被归类，且有Initial,不保留
    # 'Sex',有'Sex_male', 'Sex_female'了，可以不保留
    #  'Age',有Age_new，可以不保留
    #  'SibSp',选择保留，生成alone和family_size
    #  'Parch',选择保留，生成alone和family_size
    # 'Ticket',任意的字符串，不能被归类，不保留
    #  'Fare',有Fare_new了，不保留
    #  'Cabin',空值太多，不保留
    #  'Embarked',有用的特征，保留
    data.drop(['PassengerId', 'Name', 'Sex', 'Ticket', "Fare", "Cabin", 'Age'], axis=1, inplace=True)  # 删除不用保留的特征

    print(data.columns)
    # 保存数据
    data.to_csv(save_path, index=False)
