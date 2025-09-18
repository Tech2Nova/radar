"""
URL configuration for radar project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path

# urlpatterns = [
#     path("admin/", admin.site.urls),
# ]
from django.urls import include, path  # 更新导入方式

import submission.views

urlpatterns = [
    path("", submission.views.index, name='index'),  # 使用path替代url
    path("submission/", include("submission.urls")),  # 包含提交模块的URL
    path('analysis/', include('radar.analysis.urls')),

]
#
# path("analysis/", include("analysis.urls")),  # 包含分析模块的URL
# path("compare/", include("compare.urls")),  # 包含比较模块的URL
# path("realtime/", include("realtime.urls")),  # 包含实时检测模块的URL
#
# path("file/<str:category>/<str:object_id>/", analysis.views.file, name='analysis.views.file'),  # 使用path参数
# path("full_memory/<str:analysis_number>/", analysis.views.full_memory_dump_file, name='full_memory'),  # 使用path参数
# path("dashboard/", include("dashboard.urls")),  # 包含仪表板模块的URL