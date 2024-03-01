from django.db import models

# Create your models here.
"""数据库中的关系表（关系型数据库）"""


class Category(models.Model):  # 类别
    category = models.PositiveIntegerField(unique=True, null=False, primary_key=True)  # Category编号
    name = models.CharField(max_length=100, null=True)  # Category名称
    video_clip_number = models.PositiveIntegerField(null=True)  # 数据集中该类有的片段数量

    def __str__(self):
        return self.name if self.name else f'category #{self.category}'


class Video_Clip(models.Model):  # 视频片段
    video_id = models.PositiveIntegerField(unique=True, null=False, primary_key=True)  # video_id
    category = models.ForeignKey(to=Category, on_delete=models.PROTECT, null=True)
    sentences = models.TextField(null=True)  # 视频描述（若有多个则只好以换行区分）

    def __str__(self):
        return self.sentences.split('\n', maxsplit=1)[0] if self.sentences \
            else str(self.video_id)
