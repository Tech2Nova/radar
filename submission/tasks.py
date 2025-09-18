# submission/tasks.py
from celery import shared_task
from .models import Task
import time

@shared_task
def process_file_task(task_id, files):
    time.sleep(10)  # 替换为实际分析逻辑
    task = Task.objects.get(task_id=task_id)
    task.status = "reported"
    task.save()