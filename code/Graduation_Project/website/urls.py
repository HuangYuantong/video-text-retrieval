from django.urls import path
from . import views

# Create your views here.
"""分析传回的url，关联到相应的view.py中函数"""

urlpatterns = [
    path('', views.home_valon),
    path('home_valon', views.home_valon, name='home_valon'),
    path('home_valoff', views.home_valoff, name='home_valoff'),
    path('video_valoff/<str:pk>', views.video_player_valoff, name='video_player_valoff'),
    path('video_valon/<str:pk>', views.video_player_valon, name='video_player_valon'),
    path('db_construct', views.db_construct, name='db_construct'),  # 数据库修改
]
