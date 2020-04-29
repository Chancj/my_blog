from django.urls import path
# 引入views.py
from .views import *

app_name = 'titanic'

urlpatterns = [
    path('pred/', pred),
    path('pred2/', pred2),
    path('index2/', index2, name="titanicindex2"),
    path('', index, name="titanicindex")
]