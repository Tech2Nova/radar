# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# 本文件是Cuckoo Sandbox的一部分 - http://www.cuckoosandbox.org
# 请参阅文件 'docs/LICENSE' 以获取复制权限。


# urlpatterns = [
#     path("", views.index, name='submission.views.index'),  # 根路径
#     path("status/<int:task_id>/", views.status, name='submission.views.status'),  # 状态路径
#     path("<int:task_id>/", views.resubmit, name='submission.views.resubmit'),  # 重新提交路径
#     path("<int:task_id>/dropped/<str:sha1>/", views.submit_dropped, name='submission.views.submit_dropped'),  # 提交丢失路径
# ]

from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path("", views.index, name='submission.views.index'),
    path('complete/', views.complete, name='submission.views.complete'),
    path('status/<uuid:task_id>/', views.status, name='submission.views.status'),
]

# Create your views here.