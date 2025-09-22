# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# 本文件是Cuckoo Sandbox的一部分 - http://www.cuckoosandbox.org
# 查看文件 "docs/LICENSE" 以获取复制权限。

from . import views
from django.urls import path  # 更新为Django 5.0.8的导入方式

urlpatterns = [
    path("", views.index, name='dashboard.views.index'),
]
