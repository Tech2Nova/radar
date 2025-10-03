# submission/tasks.py
from celery import shared_task
from .models import Task
import logging
import time

logger = logging.getLogger(__name__)

@shared_task
def process_file_task(task_id, files):
    logger.info(f"Processing task {task_id} with files: {files}")
    try:
        # 模拟文件处理
        time.sleep(10)  # 替换为实际分析逻辑
        task = Task.objects.get(task_id=task_id)
        task.status = "reported"
        task.save()
        logger.info(f"Task {task_id} status updated to reported")
    except Task.DoesNotExist:
        logger.error(f"Task {task_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        raise
