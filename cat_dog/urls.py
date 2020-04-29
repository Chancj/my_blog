from django.urls import path
# 引入views.py
from .views import *

app_name = 'cat_dog'

urlpatterns = [
    path('upload/', upload),
    path('pred/', pred),
    path('', index, name="catdogindex")
]