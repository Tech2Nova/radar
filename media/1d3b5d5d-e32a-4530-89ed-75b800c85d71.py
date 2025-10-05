import os
from celery import Celery

# 设置 Django 设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'radar.settings')

# 创建 Celery 实例
app = Celery('radar')

# 从 Django 设置加载 Celery 配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现任务（从 INSTALLED_APPS 中）
app.autodiscover_tasks()

# 可选：调试任务
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
