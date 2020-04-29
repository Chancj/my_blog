from django.shortcuts import render
# 基础版模型
from .ml import MachineLearn
# 升级版模型
from .ml_new import MachineLearn_new
from django.http import JsonResponse


# Create your views here.
def index(request):
    return render(request, 'titanic/index.html')


def index2(request):
    return render(request, 'titanic/index2.html')


def pred(request):
    """
    泰坦尼克号基础版的业务逻辑代码
    :param request:
    :return:
    """
    sex = request.POST.get("sex", None)
    alone = request.POST.get("alone", None)
    age = request.POST.get("age", None)
    fare = request.POST.get("fare", None)
    # 算法选择
    logic_select = request.POST.get("logic_select", None)
    # 调用模型进行训练
    ml = MachineLearn(sex, age, fare, alone)
    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml.KNN()
    elif logic_select == "LogicRegression":
        res_pred = ml.LogicRegression()
    elif logic_select == "DecisionTree":
        res_pred = ml.DecisionTree()
    elif logic_select == "RandomForest":
        res_pred = ml.RandomForest()
    elif logic_select == "SVM":
        res_pred = ml.SVM()
    elif logic_select == "Bagging":
        res_pred = ml.Bagging()
    elif logic_select == "Adaboost":
        res_pred = ml.Adaboost()

    return JsonResponse({"msg": res_pred[0], "acc": res_pred[1]})


def pred2(request):
    """
    泰坦尼克号升级版的业务逻辑代码
    :param request:
    :return:
    """
    sex = request.POST.get("sex", None)
    initial = request.POST.get("initial", None)
    age = request.POST.get("age", None)
    sibsp = request.POST.get("sibsp", None)
    parch = request.POST.get("parch", None)
    fare = request.POST.get("fare", None)
    embarked = request.POST.get("embarked", None)
    pclass = request.POST.get("pclass", None)
    logic_select = request.POST.get("logic_select", None)

    ml = MachineLearn_new(sex, initial, age, sibsp, parch, fare, embarked, pclass)
    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml.KNN()
    elif logic_select == "LogicRegression":
        res_pred = ml.LogicRegression()
    elif logic_select == "DecisionTree":
        res_pred = ml.DecisionTree()
    elif logic_select == "RandomForest":
        res_pred = ml.RandomForest()
    elif logic_select == "SVM":
        res_pred = ml.SVM()
    elif logic_select == "Bagging":
        res_pred = ml.Bagging()
    elif logic_select == "Adaboost":
        res_pred = ml.Adaboost()

    return JsonResponse({"msg": res_pred[0], "acc": res_pred[1]})
