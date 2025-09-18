from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.urls import reverse
from .models import Task
from .tasks import process_file_task  # 导入 Celery 任务
import logging

logger = logging.getLogger(__name__)


def index(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    logger.info(f"Request method: {request.method}, Headers: {request.headers}")

    if request.method == "POST":
        logger.info(f"POST data: {request.POST}")
        logger.info(f"FILES data: {request.FILES}")
        category = request.POST.get("category")
        logger.info(f"Category: {category}")

        fs = FileSystemStorage()
        if category == "file" and request.FILES.getlist("sample"):
            samples = request.FILES.getlist("sample")
            MAX_FILES = getattr(settings, 'MAX_FILES', 10)
            if len(samples) > MAX_FILES:
                return JsonResponse({"error": f"上传文件数量超过限制（最多 {MAX_FILES} 个)。"}, status=400)
            saved_files = []
            try:
                for sample in samples:
                    if not sample.size:
                        return JsonResponse({"error": "您上传了一个空文件。"}, status=400)
                    if sample.size > getattr(settings, 'MAX_UPLOAD_SIZE', 5 * 1024 * 1024):
                        return JsonResponse({"error": "您上传的文件超过了最大允许上传大小（5MB）。"}, status=400)
                    filename = fs.save(sample.name, sample)
                    saved_files.append(filename)
                    logger.info(f"File saved at: {fs.path(filename)}")

                sample_id = request.POST.get("sample_id")
                if sample_id:
                    logger.info(f"Resubmitting sample ID: {sample_id}")

                # 创建 Task 实例
                task = Task.objects.create(
                    files=saved_files,
                    sample_id=sample_id
                )
                task_id = task.task_id

                logger.info(f"Created task {task_id} with files: {saved_files}")

                # 启动 Celery 任务
                process_file_task.delay(str(task.task_id), saved_files)
                return JsonResponse({"task_id": str(task_id)})
            except Exception as e:
                logger.error(f"Error processing file upload: {str(e)}")
                return JsonResponse({"error": f"文件上传失败：{str(e)}"}, status=500)

        return JsonResponse({"error": f"无效的提交类型。Category: {category}, Files: {bool(request.FILES.getlist('sample'))}"}, status=400)

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


def status(request, task_id):
    try:
        task = Task.objects.get(task_id=task_id)
    except Task.DoesNotExist:
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        return JsonResponse({"error": "指定的任务似乎不存在。"}, status=404) if is_ajax else render(
            request, "error.html", {"error": "指定的任务似乎不存在。"}
        )

    if task.status == "reported":
        redirect_url = reverse("analysis.views.report", kwargs={"task_id": task_id})
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        return JsonResponse({"status": "reported", "redirect": redirect_url}) if is_ajax else redirect(
            "analysis.views.report", task_id=task_id
        )

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    return JsonResponse({"status": task.status, "task_id": str(task_id)}) if is_ajax else render(
        request, "submission/status.html", {"status": task.status, "task_id": task_id}
    )