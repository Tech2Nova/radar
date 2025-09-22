# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.

import sys
import time

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_safe

# 将CUCKOO_PATH添加到系统路径
sys.path.append(settings.CUCKOO_PATH)


def timestamp(dt):
    """返回datetime对象的时间戳。"""
    if not dt:
        return None
    return time.mktime(dt.timetuple())

@require_safe
def index(request):


    # 渲染模板并返回报告
    return render(request, "dashboard/index.html" )
