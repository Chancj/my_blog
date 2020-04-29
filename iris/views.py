from django.shortcuts import render
from .ml import MachineLearn
from django.http import JsonResponse


# Create your views here.

def index(request):
    return render(request, 'iris/index.html')


def linear_pred(request):
    petal_width = float(request.POST.get("petal_width", None))
    linear_select = request.POST.get("linear_select", None)
    ml = MachineLearn()
    res = ""
    if linear_select == "LinearRegression":
        res = ml.LineRegression(petal_width)
    elif linear_select == "PolyRegression":
        res = ml.PolyRegresson(petal_width)
    # print("*"*50,res)
    return JsonResponse({"msg": res})


def pred(request):
    # names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    petal_width2 = float(request.POST.get("petal_width2", None))
    petal_length = float(request.POST.get("petal_length", None))
    sepal_width = float(request.POST.get("sepal_width", None))
    sepal_length = float(request.POST.get("sepal_length", None))
    logic_select = request.POST.get("logic_select", None)

    ml = MachineLearn()
    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml.KNN(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])
    elif logic_select == "LogicRegression":
        res_pred = ml.LogsticRegression(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])
    elif logic_select == "DecisionTree":
        res_pred = ml.DecideTree(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])
    elif logic_select == "RandomForest":
        res_pred = ml.RandomForest(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])
    elif logic_select == "SVM":
        res_pred = ml.SVM(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])
    elif logic_select == "Cluster":
        res_pred = ml.Cluster(pred=[[sepal_length, sepal_width, petal_length, petal_width2]])

    return JsonResponse({"msg": res_pred[0], "acc": res_pred[1]})
