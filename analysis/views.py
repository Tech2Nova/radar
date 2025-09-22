# coding=utf-8
# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.http import require_safe
from django.views.decorators.csrf import csrf_exempt

@require_safe
def index(request):
    return render(request, "analysis/index.html", {})

@require_safe
def pending(request):
    return render(request, "analysis/pending.html", {})

@require_safe
def chunk(request, task_id, pid, pagenum):
    return render(request, "analysis/behavior/_chunk.html", {})

@require_safe
def filtered_chunk(request, task_id, pid, category):
    return render(request, "analysis/behavior/_chunk.html", {})

@csrf_exempt
def search_behavior(request, task_id):
    return render(request, "analysis/behavior/_search_results.html", {})

@require_safe
def report(request, task_id):
    return render(request, "analysis/report.html", {})

@require_safe
def latest_report(request):
    return render(request, "analysis/report.html", {})

@require_safe
def file(request, category, object_id):
    return render(request, "error.html", {})

@require_safe
def moloch(request, **kwargs):
    return render(request, "error.html", {})

@require_safe
def full_memory_dump_file(request, analysis_number):
    return render(request, "error.html", {})

@csrf_exempt
def search(request):
    return render(request, "analysis/search.html", {})

@require_safe
def remove(request, task_id):
    return render(request, "success.html", {})

@require_safe
def pcapstream(request, task_id, conntuple):
    return render(request, "standalone_error.html", {})

@require_safe
def export_analysis(request, task_id):
    return render(request, "analysis/export.html", {})

@csrf_exempt
def export(request, task_id):
    return HttpResponse(content_type="application/zip")

@csrf_exempt
def import_analysis(request):
    return render(request, "analysis/import.html", {})

@require_safe
def chat_view(request, id):
    return render(request, "overview/index.html", {'analysis_id': id})

@csrf_exempt
def get_response(request, id):
    return JsonResponse({})

