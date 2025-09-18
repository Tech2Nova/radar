# coding=utf-8
# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.

import sys
import re
import os
import json
import urllib.parse  # 修改为 urllib.parse
import zipfile
from io import BytesIO  # 修改为 BytesIO

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_safe
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
from django.db import models

import pymongo
from bson.objectid import ObjectId
from django.core.exceptions import PermissionDenied, ObjectDoesNotExist
from gridfs import GridFS

sys.path.append(settings.CUCKOO_PATH)

from lib.cuckoo.core.database import Database, TASK_PENDING, TASK_COMPLETED
from lib.cuckoo.common.utils import store_temp_file, versiontuple
from lib.cuckoo.common.constants import CUCKOO_ROOT
import modules.processing.network as network

results_db = settings.MONGO
fs = GridFS(results_db)

@require_safe
def index(request):
    db = Database()
    tasks_files = db.list_tasks(limit=50, category="file", not_status=TASK_PENDING)
    tasks_urls = db.list_tasks(limit=50, category="url", not_status=TASK_PENDING)

    analyses_files = []
    analyses_urls = []

    if tasks_files:
        for task in tasks_files:
            new = task.to_dict()
            new["sample"] = db.view_sample(new["sample_id"]).to_dict()

            filename = os.path.basename(new["target"])
            new.update({"filename": filename})

            if db.view_errors(task.id):
                new["errors"] = True

            analyses_files.append(new)

    if tasks_urls:
        for task in tasks_urls:
            new = task.to_dict()

            if db.view_errors(task.id):
                new["errors"] = True

            analyses_urls.append(new)

    return render(request, "analysis/index.html", {
        "files": analyses_files,
        "urls": analyses_urls,
    })

@require_safe
def pending(request):
    db = Database()
    tasks = db.list_tasks(status=TASK_PENDING)

    pending = []
    for task in tasks:
        pending.append(task.to_dict())

    return render(request, "analysis/pending.html", {
        "tasks": pending,
    })

@require_safe
def chunk(request, task_id, pid, pagenum):
    try:
        pid, pagenum = int(pid), int(pagenum) - 1
    except ValueError:  # 捕获特定异常
        raise PermissionDenied

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'
    
    if not is_ajax:
        return JsonResponse({'error': 'This is not an AJAX request.'}, status=400)

    record = results_db.analysis.find_one(
        {
            "info.id": int(task_id),
            "behavior.processes.pid": pid
        },
        {
            "behavior.processes.pid": 1,
            "behavior.processes.calls": 1
        }
    )

    if not record:
        raise ObjectDoesNotExist

    process = next((pdict for pdict in record["behavior"]["processes"] if pdict["pid"] == pid), None)

    if not process:
        raise ObjectDoesNotExist


    # 假设 process 是一个包含 calls 的字典
    if 0 <= pagenum < len(process["calls"]):
        objectid = process["calls"][pagenum]

        # 确保 objectid 是 ObjectId 或字符串
        if isinstance(objectid, list):
            # 处理列表情况，取第一个元素
            if len(objectid) > 0:
                objectid = objectid[0]  # 取第一个 ObjectId
            else:
                raise Http404("Object ID not found.")
        
        # 确保 objectid 是 ObjectId 类型或字符串
        if isinstance(objectid, ObjectId):
            # 直接使用 objectid
            chunk = results_db.calls.find_one({"_id": objectid})
        elif isinstance(objectid, str):
            # 如果是字符串，则转换为 ObjectId
            chunk = results_db.calls.find_one({"_id": ObjectId(objectid)})
        else:
            raise TypeError(f"objectid must be an ObjectId or a string, got {type(objectid)}: {objectid}")

        # 确保 chunk 存在并处理 calls
        if chunk and "calls" in chunk:
            for idx, call in enumerate(chunk["calls"]):
                call["id"] = pagenum * 100 + idx
        else:
            chunk = dict(calls=[])
    else:
        chunk = dict(calls=[])


    return render(request, "analysis/behavior/_chunk.html", {
        "chunk": chunk,
    })

@require_safe
def filtered_chunk(request, task_id, pid, category):
    """过滤特定类别的调用。
    @param task_id: cuckoo 任务 ID
    @param pid: 进程 ID
    @param category: 调用类别
    """

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

    if not is_ajax:
        return JsonResponse({'error': 'This is not an AJAX request.'}, status=400)

    # 查找与 PID 相关的调用
    record = results_db.analysis.find_one(
        {
            "info.id": int(task_id),
            "behavior.processes.pid": int(pid),
        },
        {
            "behavior.processes.pid": 1,
            "behavior.processes.calls": 1,
        }
    )

    if not record:
        raise ObjectDoesNotExist

    # 从响应集合中提取与进程相关的嵌入文档
    process = next((pdict for pdict in record["behavior"]["processes"] if pdict["pid"] == int(pid)), None)

    if not process:
        raise ObjectDoesNotExist

    # 创建空的进程字典以供 AJAX 视图使用
    filtered_process = {
        "pid": pid,
        "calls": [],
    }

    # 填充字典，从所有调用中获取数据并仅选择适当的类别
    for call in process["calls"]:
        chunk = results_db.calls.find_one({"_id": call})
        for call in chunk["calls"]:
            if call["category"] == category:
                filtered_process["calls"].append(call)

    return render(request, "analysis/behavior/_chunk.html", {
        "chunk": filtered_process,
    })

@csrf_exempt
def search_behavior(request, task_id):
    if request.method != "POST":
        raise PermissionDenied

    query = request.POST.get("search")
    query = re.compile(query, re.I)
    results = []

    # 获取分析报告
    record = results_db.analysis.find_one(
        {
            "info.id": int(task_id),
        }
    )

    # 遍历每个进程
    for process in record["behavior"]["processes"]:
        process_results = []

        chunks = results_db.calls.find({
            "_id": {"$in": process["calls"]}
        })

        index = -1
        for chunk in chunks:
            for call in chunk["calls"]:
                index += 1

                if query.search(call["api"]):
                    call["id"] = index
                    process_results.append(call)
                    continue

                for key, value in call["arguments"].items():
                    if query.search(key):
                        call["id"] = index
                        process_results.append(call)
                        break

                    if isinstance(value, str) and query.search(value):  # 修改为 str
                        call["id"] = index
                        process_results.append(call)
                        break

                    if isinstance(value, (tuple, list)):
                        for arg in value:
                            if not isinstance(arg, str):  # 修改为 str
                                continue

                            if query.search(arg):
                                call["id"] = index
                                process_results.append(call)
                                break
                        else:
                            continue
                        break

        if process_results:
            results.append({
                "process": process,
                "signs": process_results
            })

    return render(request, "analysis/behavior/_search_results.html", {
        "results": results,
    })

@require_safe
def report(request, task_id):
    # 从数据库中查找指定任务的分析报告
    report = results_db.analysis.find_one({"info.id": int(task_id)}, sort=[("_id", pymongo.DESCENDING)])

    if not report:
        return render(request, "error.html", {
            "error": "指定的分析不存在",
        })

    # 创建按域和 IP 的 DNS 信息字典
    if "network" in report and "domains" in report["network"]:
        domainlookups = {i["domain"]: i["ip"] for i in report["network"]["domains"]}
        iplookups = {i["ip"]: i["domain"] for i in report["network"]["domains"]}
        for i in report["network"]["dns"]:
            for a in i["answers"]:
                iplookups[a["data"]] = i["request"]
    else:
        domainlookups = {}
        iplookups = {}


    # Ensure virustotal attribute exists and has positives field
    virustotal_info = report.get("virustotal", {})
    scans = virustotal_info.get("scans", {})

    antivirus = {}
    selected_antivirus = ['CrowdStrike', 'Kingsoft', 'Tencent', 'Alibaba', 'McAfee', 'Avast', 'Baidu']


    for engine, result in scans.items():
        if engine in selected_antivirus:
            antivirus[engine] = result

    # detection = report.get("detection", {})


    return render(request, "analysis/report.html", {
        "analysis": report,
        "domainlookups": domainlookups,
        "iplookups": iplookups,
        "antivirus": antivirus
    })

@require_safe
def latest_report(request):
    rep = results_db.analysis.find_one({}, sort=[("_id", pymongo.DESCENDING)])
    return report(request, rep["info"]["id"] if rep else 0)

@require_safe
def file(request, category, object_id):
    file_item = fs.get(ObjectId(object_id))

    if file_item:
        # 以 sha256_originalfilename 格式组合文件名
        file_name = f"{file_item.sha256}_{file_item.filename}"

        # 处理 gridfs 错误，如果缺少 contentType 字段
        content_type = getattr(file_item, 'contentType', "application/octet-stream")

        response = HttpResponse(file_item.read(), content_type=content_type)
        response["Content-Disposition"] = f"attachment; filename={file_name}"

        return response
    else:
        return render(request, "error.html", {
            "error": "文件未找到",
        })

moloch_mapper = {
    "ip": "ip == %s",
    "host": "host == %s",
    "src_ip": "ip == %s",
    "src_port": "port == %s",
    "dst_ip": "ip == %s",
    "dst_port": "port == %s",
    "sid": 'tags == "sid:%s"',
}

@require_safe
def moloch(request, **kwargs):
    # 检查 Moloch 是否启用
    if not settings.MOLOCH_ENABLED:
        return render(request, "error.html", {
            "error": "后续扩展:Moloch: 网络流量收集与分析系统",
        })

    query = []
    # 构建查询条件
    for key, value in kwargs.items():
        if value and value != "None":
            query.append(moloch_mapper[key] % value)

    # 获取主机名
    hostname = request.get_host().split(":")[0] if ":" in request.get_host() else request.get_host()


    url = "https://%s:8005/?%s" % (
        settings.MOLOCH_HOST or hostname,
        urllib.urlencode({
            "date": "-1",
            "expression": " && ".join(query),
        }),
    )

    
    return redirect(url)  # 重定向到构建的 URL

@require_safe
def full_memory_dump_file(request, analysis_number):
    file_path = os.path.join(CUCKOO_ROOT, "storage", "analyses", str(analysis_number), "memory.dmp")
    if os.path.exists(file_path):
        content_type = "application/octet-stream"
        response = HttpResponse(open(file_path, "rb").read(), content_type=content_type)
        response["Content-Disposition"] = "attachment; filename=memory.dmp"
        return response
    else:
        return render(request, "error.html", {
            "error": "文件未找到",
        })

def _search_helper(obj, k, value):
    r = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            r += _search_helper(v, k, value)

    if isinstance(obj, (tuple, list)):
        for v in obj:
            r += _search_helper(v, k, value)

    if isinstance(obj, str):  # 修改为 str
        if re.search(value, obj, re.I):
            r.append((k, obj))

    return r

@csrf_exempt
def search(request):
    """使用 ElasticSearch 作为后端的新搜索 API。"""
    if not settings.ELASTIC:
        return render(request, "error.html", {
            "error": "ElasticSearch 未启用，因此无法进行全局搜索。",
        })

    if request.method == "GET":
        return render(request, "analysis/search.html")

    value = request.POST["search"]

    match_value = ".*".join(re.split("[^a-zA-Z0-9]+", value.lower()))

    r = settings.ELASTIC.search(body={
        "query": {
            "query_string": {
                "query": f'"{value}"*',
            },
        },
    })

    analyses = []
    for hit in r["hits"]["hits"]:
        # 查找此命中的实际匹配并限制为 8 个匹配项
        matches = _search_helper(hit, "none", match_value)
        if not matches:
            continue

        analyses.append({
            "task_id": hit["_index"].split("-")[-1],
            "matches": matches[:16],
            "total": max(len(matches) - 16, 0),
        })

    if request.POST.get("raw"):
        return render(request, "analysis/search_results.html", {
            "analyses": analyses,
            "term": request.POST["search"],
        })

    return render(request, "analysis/search.html", {
        "analyses": analyses,
        "term": request.POST["search"],
        "error": None,
    })


@require_safe
def remove(request, task_id):
    """删除分析。
    @todo: 从存储中删除文件夹。
    """
    analyses = list(results_db.analysis.find({"info.id": int(task_id)}))  # 将游标转换为列表

    # 检查是否找到多个具有相同 ID 的分析，例如如果手动运行了 process.py
    if len(analyses) > 1:
        message = (
            "删除了多个具有此 ID 的任务，感谢您的支持 "
            "(指定的分析在 mongo 中存在多次)。"
        )
    elif len(analyses) == 1:
        message = "任务已删除，感谢您的支持。"
    else:
        return render(request, "error.html", {
            "error": "指定的分析不存在",
        })

    for analysis in analyses:
        # 如果未使用，则删除样本
        if "file_id" in analysis["target"]:
            if results_db.analysis.count_documents({"target.file_id": ObjectId(analysis["target"]["file_id"])}) == 1:
                fs.delete(ObjectId(analysis["target"]["file_id"]))

        # 删除截图
        for shot in analysis["shots"]:
            if results_db.analysis.count_documents({"shots": ObjectId(shot)}) == 1:
                fs.delete(ObjectId(shot))

        # 删除网络 pcap
        if "pcap_id" in analysis["network"] and results_db.analysis.count_documents({"network.pcap_id": ObjectId(analysis["network"]["pcap_id"])}) == 1:
            fs.delete(ObjectId(analysis["network"]["pcap_id"]))

        # 删除排序后的 pcap
        if "sorted_pcap_id" in analysis["network"] and results_db.analysis.count_documents({"network.sorted_pcap_id": ObjectId(analysis["network"]["sorted_pcap_id"])}) == 1:
            fs.delete(ObjectId(analysis["network"]["sorted_pcap_id"]))

        # 删除 mitmproxy 转储
        if "mitmproxy_id" in analysis["network"] and results_db.analysis.count_documents({"network.mitmproxy_id": ObjectId(analysis["network"]["mitmproxy_id"])}) == 1:
            fs.delete(ObjectId(analysis["network"]["mitmproxy_id"]))

        # 删除丢失的文件
        for drop in analysis["dropped"]:
            if "object_id" in drop and results_db.analysis.count_documents({"dropped.object_id": ObjectId(drop["object_id"])}) == 1:
                fs.delete(ObjectId(drop["object_id"]))

        # 删除调用
        for process in analysis.get("behavior", {}).get("processes", []):
            for call in process["calls"]:
                results_db.calls.delete_one({"_id": ObjectId(call)})  # 修改为 delete_one

        # 删除分析数据
        results_db.analysis.delete_one({"_id": ObjectId(analysis["_id"])})  # 修改为 delete_one

    # 从 SQL 数据库中删除
    db = Database()
    db.delete_task(task_id)

    return render(request, "success.html", {
        "message": message,
    })


@require_safe
def pcapstream(request, task_id, conntuple):
    """获取与特定连接相关的任务 PCAP 中的数据包。
    这是可能的，因为我们在处理期间对 PCAP 进行了排序，并记住了每个流的偏移量。
    """
    src, sport, dst, dport, proto = conntuple.split(",")
    sport, dport = int(sport), int(dport)

    conndata = results_db.analysis.find_one(
        {
            "info.id": int(task_id),
        },
        {
            "network.tcp": 1,
            "network.udp": 1,
            "network.sorted_pcap_id": 1,
        },
        sort=[("_id", pymongo.DESCENDING)]
    )

    if not conndata:
        return render(request, "standalone_error.html", {
            "error": "指定的分析不存在",
        })

    try:
        connlist = conndata["network"]["udp"] if proto == "udp" else conndata["network"]["tcp"]

        conns = [i for i in connlist if (i["sport"], i["dport"], i["src"], i["dst"]) == (sport, dport, src, dst)]
        stream = conns[0]
        offset = stream["offset"]
    except (IndexError, KeyError):  # 捕获特定异常
        return render(request, "standalone_error.html", {
            "error": "找不到请求的流",
        })

    try:
        fobj = fs.get(conndata["network"]["sorted_pcap_id"])
        setattr(fobj, "fileno", lambda: -1)
    except:
        return render(request, "standalone_error.html", {
            "error": "所需的排序 PCAP 不存在",
        })

    packets = list(network.packets_for_stream(fobj, offset))
    return HttpResponse(json.dumps(packets), content_type="application/json")

def export_analysis(request, task_id):
    if request.method == "POST":
        return export(request, task_id)

    report = results_db.analysis.find_one(
        {"info.id": int(task_id)}, sort=[("_id", pymongo.DESCENDING)]
    )
    if not report:
        return render(request, "error.html", {
            "error": "指定的分析不存在",
        })

    if "analysis_path" not in report.get("info", {}):
        return render(request, "error.html", {
            "error": "该分析是在导出功能与 Cuckoo 集成之前创建的，因此无法导出此任务 "
                     "(要导出此分析，请重新处理其报告)。"
        })

    analysis_path = report["info"]["analysis_path"]

    # 查找此分析的所有目录/结果
    dirs, files = [], []
    for filename in os.listdir(analysis_path):
        path = os.path.join(analysis_path, filename)
        if os.path.isdir(path):
            dirs.append((filename, len(os.listdir(path))))
        else:
            files.append(filename)

    return render(request, "analysis/export.html", {
        "analysis": report,
        "dirs": dirs,
        "files": files,
    })

def export(request, task_id):
    taken_dirs = request.POST.getlist("dirs")
    taken_files = request.POST.getlist("files")
    if not taken_dirs and not taken_files:
        return render(request, "error.html", {
            "error": "请至少选择一个目录或文件进行导出。"
        })

    report = results_db.analysis.find_one(
        {"info.id": int(task_id)}, sort=[("_id", pymongo.DESCENDING)]
    )
    if not report:
        return render(request, "error.html", {
            "error": "指定的分析不存在",
        })

    path = report["info"]["analysis_path"]

    # 创建一个 analysis.json 文件，包含有关此分析的基本信息。
    # 此信息作为导入任务时的元数据。
    analysis_path = os.path.join(path, "analysis.json")
    with open(analysis_path, "w") as outfile:
        report["target"].pop("file_id", None)
        json.dump({"target": report["target"]}, outfile, indent=4)

    f = BytesIO()  # 修改为 BytesIO

    # 创建一个 zip 文件，包含任务的选定文件和目录。
    zf = zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED)

    for dirname, subdirs, files in os.walk(path):
        if os.path.basename(dirname) == task_id:
            for filename in files:
                if filename in taken_files:
                    zf.write(os.path.join(dirname, filename), filename)
        if os.path.basename(dirname) in taken_dirs:
            for filename in files:
                zf.write(os.path.join(dirname, filename),
                         os.path.join(os.path.basename(dirname), filename))

    zf.close()

    response = HttpResponse(f.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = f"attachment; filename={task_id}.zip"
    return response

def import_analysis(request):
    if request.method == "GET":
        return render(request, "analysis/import.html")

    db = Database()
    task_ids = []
    analyses = request.FILES.getlist("sample")

    for analysis in analyses:
        if not analysis.size:
            return render(request, "error.html", {
                "error": "您上传了一个空分析。",
            })

        # if analysis.size > settings.MAX_UPLOAD_SIZE:
            # return render(request, "error.html", {
            #     "error": "您上传的文件超过了最大允许上传大小。",
            # })

        if not analysis.name.endswith(".zip"):
            return render(request, "error.html", {
                "error": "您上传的分析不是 .zip 文件。",
            })

        zf = zipfile.ZipFile(analysis)

        # 根据 Python 文档，我们必须确保没有不正确的文件名。
        for filename in zf.namelist():
            if filename.startswith("/") or ".." in filename or ":" in filename:
                return render(request, "error.html", {
                    "error": "压缩文件包含不正确的文件名，请提供合法的 .zip 文件。",
                })

        if "analysis.json" in zf.namelist():
            analysis_info = json.loads(zf.read("analysis.json"))
        elif "binary" in zf.namelist():
            analysis_info = {
                "target": {
                    "category": "file",
                },
            }
        else:
            analysis_info = {
                "target": {
                    "category": "url",
                    "url": "unknown",
                },
            }

        category = analysis_info["target"]["category"]

        if category == "file":
            binary = store_temp_file(zf.read("binary"), "binary")

            if os.path.isfile(binary):
                task_id = db.add_path(file_path=binary,
                                      package="",
                                      timeout=0,
                                      options="",
                                      priority=0,
                                      machine="",
                                      custom="",
                                      memory=False,
                                      enforce_timeout=False,
                                      tags=None)
                if task_id:
                    task_ids.append(task_id)

        elif category == "url":
            url = analysis_info["target"]["url"]
            if not url:
                return render(request, "error.html", {
                    "error": "您指定了一个无效的 URL！",
                })

            task_id = db.add_url(url=url,
                                 package="",
                                 timeout=0,
                                 options="",
                                 priority=0,
                                 machine="",
                                 custom="",
                                 memory=False,
                                 enforce_timeout=False,
                                 tags=None)
            if task_id:
                task_ids.append(task_id)

        if not task_id:
            continue

        # 提取与此分析相关的所有文件。这可能需要一些黑客技术，具体取决于 Web 界面运行的用户/组。
        analysis_path = os.path.join(
            CUCKOO_ROOT, "storage", "analyses", f"{task_id}"
        )

        if not os.path.exists(analysis_path):
            os.mkdir(analysis_path)

        zf.extractall(analysis_path)

        # 我们将此分析设置为已完成，以便它将被自动处理
        # （假设 process.py / process2.py 正在运行）。
        db.set_status(task_id, TASK_COMPLETED)

    if task_ids:
        return render(request, "submission/complete.html", {
            "tasks": task_ids,
            "baseurl": request.build_absolute_uri("/")[:-1],
        })


from django.shortcuts import render
from django.http import JsonResponse
import openai  # 确保你已经安装了openai库
import requests
from datetime import datetime

# 设置OpenAI API密钥
openai.api_key = 'sk-ZnydsaV1KLgT214bA65bAe8eC44043F4B837C2E3Dd2e96C0'

# 自定义的 JSON 编码器，用于处理 ObjectId 和 datetime
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# 渲染聊天页面
def chat_view(request, id):
    return render(request, 'overview/index.html', {'analysis_id': id})

# 处理聊天请求
def get_response(request, id):
    user_message = request.GET.get('message')
    model = request.GET.get('model', 'MalwareGPT')  # 默认使用gpt4模型
    
    if not user_message:
        return JsonResponse({'error': '请输入您的问题'}, status=400)

    try:
        report = results_db.analysis.find_one(
            {"info.id": int(id)}, sort=[("_id", pymongo.DESCENDING)]
        )
        
        if not report:
            return JsonResponse({'error': '未找到相关分析报告'}, status=404)
            
        # 移除不必要的字段
        for field in ['malimg', 'static', 'debug', 'screenshots', 'target', 'metadata', 'dropped', 
                     'procmemory', 'virustotal', 'detection', 'behavior', 'network', 'strings', 
                     '_id', 'object_id', 'shots']:
            report.pop(field, None)
            
        report_str = json.dumps(report, ensure_ascii=False, cls=JSONEncoder)

        # 设置代理和超时
        proxies = {
            'http': 'socks5://192.168.0.33:7891',
            'https': 'socks5://192.168.0.33:7891'
        }
        timeout = 60  # 增加超时时间到60秒

        # 前置提示词
        pre_prompt = (
            f"You are ChatHawk, a large language model trained by Torrk, Below is a malware analysis report. "
            f"You need to read this report carefully and then answer the questions: {report_str}.\n"
            "Knowledge cutoff: 2021-09\n"
            "Current date: [current date]\n"
            "Language: Chinese\n"
        )

        # pre_prompt = (
        #     f"你是拥有丰富恶意软件专业领域知识的DeepSeek大语言模型。接下来将为你提供一份恶意软件分析报告, 你需要仔细研读这份报告, 随后精准且全面地回答基于该报告提出的各类问题。"
        #     f"这些问题可能涵盖恶意软件的基本信息（如类型、名称、动静态特征等）、其造成的危害、检测方法、应对措施以及防御策略等。"
        #     f"请基于报告内容进行作答, 以下是详细的恶意软件分析报告： {report_str}.\n"
        #     "回答要求：\n"
        #     "1. 回答需使用中文，表述务必清晰流畅、逻辑严谨，以确保回答的专业性和准确性。\n"
        #     "2. 若问题包含多个方面，需按照要点进行清晰的分点阐述，使回答层次分明、条理清晰。\n"
        #     "3. 对于分析类问题，要给出明确的结论，并提供合理且充分的依据，增强回答的可信度和说服力。\n"
        #     "4. 若报告中未提及相关内容，需明确说明报告中未提供该部分信息。\n"
        #     "知识截止日期: 2021 - 09\n"
        #     "当前日期：[current date]\n"
        #     "语言：中文\n"
        # )


        if model == 'gpt4':
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key="sk-zD9HhwpnmUDrWsP1Ey5UklfrwgPD2BHuj659OC3POsNFsbKU",
                    base_url="https://api.chatanywhere.tech/v1"
                )
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {'role': 'system', 'content': pre_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    stream=True
                )
                
                # 使用Server-Sent Events进行流式响应
                from django.http import StreamingHttpResponse
                
                def generate_response():
                    try:
                        for chunk in completion:
                            if chunk.choices[0].delta.content is not None:
                                yield f"data: {chunk.choices[0].delta.content}\n\n"
                    except Exception as e:
                        yield f"data: \n错误: {str(e)}\n\n"
                    yield "data: [DONE]\n\n"
                
                response = StreamingHttpResponse(generate_response(), content_type='text/event-stream')
                response['Cache-Control'] = 'no-cache'
                response['X-Accel-Buffering'] = 'no'
                return response
            except Exception as e:
                error_msg = str(e)
                if 'Service Unavailable' in error_msg:
                    return JsonResponse({'error': 'GPT-4服务暂时不可用，请检查API配置或稍后重试'}, status=503)
                elif 'timeout' in error_msg.lower():
                    return JsonResponse({'error': 'API请求超时，请稍后重试'}, status=504)
                elif 'connect' in error_msg.lower():
                    return JsonResponse({'error': '网络连接错误，请检查网络设置'}, status=503)
                else:
                    return JsonResponse({'error': f'API请求失败: {error_msg}'}, status=503)
        elif model == 'deepseek':
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key="sk-f71ee0570f0f416eb7872a5937a468c6",
                    base_url="https://api.deepseek.com/"
                )
                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": pre_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    stream=True
                )
                from django.http import StreamingHttpResponse
                
                def generate_response():
                    try:
                        for chunk in completion:
                            if chunk.choices[0].delta.content is not None:
                                yield f"data: {chunk.choices[0].delta.content}\n\n"
                    except Exception as e:
                        yield f"data: \n错误: {str(e)}\n\n"
                    yield "data: [DONE]\n\n"
                
                response = StreamingHttpResponse(generate_response(), content_type='text/event-stream')
                response['Cache-Control'] = 'no-cache'
                response['X-Accel-Buffering'] = 'no'
                return response
            except Exception as e:
                return JsonResponse({'error': f'Deepseek API调用失败: {str(e)}'}, status=503)
    except Exception as e:
        return JsonResponse({'error': f'服务器内部错误: {str(e)}'}, status=500)
    
    