from django.urls import path
# 引入views.py
from . import views

# from .views import *

app_name = 'titanic'

urlpatterns = [
    # 文章列表
    # path('article-list/', views.article_list, name='article_list'),
    path('pred/', views.pred),
    path('pred2/', views.pred2),
    path('index2/', views.index2, name="titanicindex2"),
    path('index/', views.index, name="titanicindex"),
]
