from django.db import models
import uuid

class Task(models.Model):
    task_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default="pending")
    result = models.IntegerField(null=True, blank=True)  # 新增：0 (benign) 或 1 (malware)
    confidence = models.FloatField(null=True, blank=True)  # 新增：置信度
    files = models.JSONField(default=list)
    sample_id = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Task {self.task_id} - {self.status}"


