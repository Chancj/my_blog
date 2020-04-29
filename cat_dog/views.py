from django.shortcuts import render
import os
from django.http import JsonResponse
from .ml import CatDog

# Create your views here.
def index(request):
    return render(request,"cat_dog/index.html")

def upload(request):
    """
    file_path_url，file_path
    :param request:
    :return:
    """
    if request.method == "POST":
        ret = {"status":False,"data":False,"error":None}
        try:

            img = request.FILES.get("img")
            FILE_PATH = os.path.abspath(os.path.dirname(__file__))+os.sep+"static"+os.sep+img.name

            FILE_PATH_URL = "/static/" + img.name  #前端展示图片
            #上传的图片写入本地
            f = open(FILE_PATH,"wb")
            for chunk in img.chunks(chunk_size=1024*1024):
                f.write(chunk)
            ret["status"] = True
            # ret["data"] = FILE_PATH_URL
        except Exception as e:
            print(e)
            ret["error"]  = e
            return JsonResponse({"file_path":"","file_path_url":"","status":ret["status"],"error":ret["error"]})
        finally:
            f.close()

        return JsonResponse({"file_path": FILE_PATH, "file_path_url": FILE_PATH_URL, "status": ret["status"], "error": ret["error"]})

def pred(request):
    file_path = request.POST.get("file_path",None)
    logic_select = request.POST.get("logic_select",None)
    ml = CatDog(file_path)
    ml.model_trian()
    if logic_select == "CNN":
        res = ml.pred_cat_dog()
    return JsonResponse({"msg":res})