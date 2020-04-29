from django.urls import path
# 引入views.py
from .views import *

app_name = 'iris'

urlpatterns = [
    path('linear_pred/', linear_pred),
    path('pred/', pred),
    path('', index, name="irisindex")
]