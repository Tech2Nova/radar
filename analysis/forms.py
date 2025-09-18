# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# 本文件是Cuckoo Sandbox的一部分 - http://www.cuckoosandbox.org
# 查看文件 'docs/LICENSE' 以获取复制权限。

from django import forms
from submission.models import Comment, Tag

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment  # 指定模型为Comment
        fields = ["message"]  # 需要包含的字段

class TagForm(forms.ModelForm):
    class Meta:
        model = Tag  # 指定模型为Tag
        fields = ["name"]  # 需要包含的字段
