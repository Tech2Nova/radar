# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# 本文件是 Cuckoo Sandbox 的一部分 - http://www.cuckoosandbox.org
# 查看文件 "docs/LICENSE" 以获取复制权限。

from . import views
from django.urls import path, re_path  # 从 django.urls 导入 path
from .views import chat_view, get_response

# re_path：复杂路由

# path： 简单路由


urlpatterns = [
    path("", views.index, name='analysis.views.index'),  # 首页
]

# path("<int:task_id>/", views.report, name='analysis.views.report'),  # 任务报告
# path("latest/", views.latest_report, name='analysis.views.latest_report'),  # 最新报告
# path("remove/<int:task_id>/", views.remove, name='analysis.views.remove'),  # 移除任务
# path("chunk/<int:task_id>/<int:pid>/<int:pagenum>/", views.chunk, name='analysis.views.chunk'),  # 数据块
# path("filtered/<int:task_id>/<int:pid>/<str:category>/", views.filtered_chunk,
#      name='analysis.views.filtered_chunk'),  # 过滤数据块
# path("search/<int:task_id>/", views.search_behavior, name='analysis.views.search_behavior'),  # 搜索行为
# path("search/", views.search, name='analysis.views.search'),  # 搜索
# path("pending/", views.pending, name='analysis.views.pending'),  # 待处理
# path("<int:task_id>/pcapstream/<str:conntuple>/", views.pcapstream, name='analysis.views.pcapstream'),  # pcap流
# re_path(
#     r"^moloch/(?P<ip>[\d\.]+)?/(?P<host>[a-zA-Z0-9-_\.]+)?/(?P<src_ip>[a-zA-Z0-9\.]+)?/(?P<src_port>\d+|None)?/(?P<dst_ip>[a-zA-Z0-9\.]+)?/(?P<dst_port>\d+|None)?/(?P<sid>\d+)?$",
#     views.moloch, name='analysis.views.moloch'),
# path("<int:task_id>/export/", views.export_analysis, name='analysis.views.export_analysis'),  # 导出分析
# path("import/", views.import_analysis, name='analysis.views.import_analysis'),  # 导入分析
# path('<int:id>/', views.chat_view, name='chat_view'),
# path('<int:id>/get_response/', views.get_response, name='get_response'),
