from django.contrib import admin
from .models import Category, Video_Clip

# Register your models here.
"""models.py中所有模型（关系表）都需要手动注册（加入数据库中）"""
admin.site.register([Category, Video_Clip, ])
