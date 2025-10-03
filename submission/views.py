from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.urls import reverse
from .models import Task
from .tasks import process_file_task  # 导入 Celery 任务
import logging
import uuid
from django.db import transaction

logger = logging.getLogger(__name__)

def index(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    logger.info(f"Request method: {request.method}, Headers: {request.headers}, POST: {request.POST}, FILES: {request.FILES}")

    if request.method == "POST":
        try:
            category = request.POST.get("category")
            logger.info(f"Category: {category}")

            fs = FileSystemStorage()
            if category == "file" and request.FILES.getlist("sample"):
                samples = request.FILES.getlist("sample")
                MAX_FILES = getattr(settings, 'MAX_FILES', 10)
                if len(samples) > MAX_FILES:
                    return JsonResponse({"error": f"上传文件数量超过限制（最多 {MAX_FILES} 个)。"}, status=400)
                saved_files = []
                for sample in samples:
                    if not sample.size:
                        return JsonResponse({"error": "您上传了一个空文件。"}, status=400)
                    if sample.size > getattr(settings, 'MAX_UPLOAD_SIZE', 5 * 1024 * 1024):
                        return JsonResponse({"error": "您上传的文件超过了最大允许上传大小（5MB）。"}, status=400)
                    try:
                        ext = sample.name.split('.')[-1] if '.' in sample.name else ''
                        filename = f"{uuid.uuid4()}.{ext}" if ext else str(uuid.uuid4())
                        filename = fs.save(filename, sample)
                        saved_files.append(filename)
                        logger.info(f"File saved at: {fs.path(filename)}")
                    except Exception as e:
                        logger.error(f"Failed to save file {sample.name}: {str(e)}")
                        return JsonResponse({"error": f"保存文件 {sample.name} 失败：{str(e)}"}, status=500)

                sample_id = request.POST.get("sample_id")
                if sample_id:
                    logger.info(f"Resubmitting sample ID: {sample_id}")

                try:
                    with transaction.atomic():  # 确保事务提交
                        task = Task.objects.create(
                            files=saved_files,
                            sample_id=sample_id
                        )
                        task_id = task.task_id
                        logger.info(f"Created task {task_id} with files: {saved_files}")
                        # 立即验证记录是否存在
                        task_verify = Task.objects.get(task_id=task_id)
                        logger.info(f"Verified task exists: {task_verify.task_id}, status: {task_verify.status}")
                except Exception as e:
                    logger.error(f"Failed to create or verify task: {str(e)}")
                    return JsonResponse({"error": f"创建任务失败：{str(e)}"}, status=500)

                try:
                    process_file_task.delay(str(task_id), saved_files)
                    logger.info(f"Celery task dispatched for task_id: {task_id}")
                except Exception as e:
                    logger.error(f"Failed to dispatch Celery task for task_id {task_id}: {str(e)}")
                    return JsonResponse({"error": f"启动分析任务失败：{str(e)}"}, status=500)

                return JsonResponse({"task_id": str(task_id)})
            else:
                return JsonResponse({"error": f"无效的提交类型。Category: {category}, Files: {bool(request.FILES.getlist('sample'))}"}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error in index view: {str(e)}")
            return JsonResponse({"error": f"服务器内部错误：{str(e)}"}, status=500)

    if is_ajax:
        return JsonResponse({"error": "仅支持 POST 请求"}, status=405)
    return render(request, "submission/index.html")


def complete(request):
    task_id = request.GET.get("task_id")
    if not task_id:
        return render(request, "error.html", {"error": "缺少任务 ID"})
    try:
        task = Task.objects.get(task_id=task_id)
        return render(request, "submission/complete.html", {
            "files": task.files,
            "files_count": len(task.files),
            "task_id": task_id,
        })
    except Task.DoesNotExist:
        return render(request, "error.html", {"error": "指定的任务不存在"})

def status_page(request):
    task_id = request.GET.get('task_id')
    if not task_id:
        return render(request, "error.html", {"error": "缺少任务 ID"})
    try:
        task = Task.objects.get(task_id=task_id)
        return render(request, "submission/status.html", {
            "task_id": task_id,
            "status": task.status,
        })
    except Task.DoesNotExist:
        return render(request, "error.html", {"error": "指定的任务不存在"})

def status(request, task_id):
    try:
        task = Task.objects.get(task_id=task_id)
    except Task.DoesNotExist:
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        return JsonResponse({"error": "指定的任务似乎不存在。"}, status=404) if is_ajax else render(
            request, "error.html", {"error": "指定的任务似乎不存在。"}
        )

    if task.status == "reported":
        redirect_url = reverse('analysis.views.index')  # 使用 reverse 动态解析 URL
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        return JsonResponse({"status": "reported", "redirect": redirect_url}) if is_ajax else redirect(redirect_url)

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    return JsonResponse({"status": task.status, "task_id": str(task_id)}) if is_ajax else render(
        request, "submission/status.html", {"status": task.status, "task_id": task_id}
    )