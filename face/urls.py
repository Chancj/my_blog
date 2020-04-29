from django.urls import path
# 引入views.py
from .views import *

app_name = 'face'

urlpatterns = [
    path("pred/", pred, name="pred"),
    path("upload/", upload, name="upload"),
    path('', index, name="faceindex"),
]