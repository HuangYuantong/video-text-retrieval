import random
import os
from typing import Union

import numpy
import torch


def set_seed(seed: int):
    """设置随机数种子"""
    random.seed(seed)  # python的随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    numpy.random.seed(seed)  # numpy的随机数种子
    torch.manual_seed(seed)  # torch的CPU
    torch.cuda.manual_seed(seed)  # torch的GPU
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def name2index(content):
    """数据集视频文件名与video_id映射规则：
    MSRVTT：’video1’->1, 1->‘video1’，0~9999
    """
    if isinstance(content, int):
        if 0 <= content <= 9999: content = f'video{content}'
    elif isinstance(content, str):
        content = int(content[5:].replace('.mp4', ''))
    else:
        raise f'video_name2index错误：{content}，type={type(content)}'
    return content


#############################################
# 计算相似度、得分、验证

def compute_metrics(sim_matrix: numpy.ndarray):
    """计算相似度矩阵的：R1、R5、R10、MedianR、MeanR"""
    sx = numpy.sort(-sim_matrix, axis=1)
    d = numpy.diag(-sim_matrix)
    d = d[:, numpy.newaxis]
    ind = sx - d
    ind = numpy.where(ind == 0)
    ind = ind[1]
    metrics = dict()
    metrics['R1'] = float(numpy.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(numpy.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(numpy.sum(ind < 10)) * 100 / len(ind)
    metrics['MedianR'] = numpy.median(ind) + 1
    metrics['MeanR'] = numpy.mean(ind) + 1
    metrics['cols'] = [int(i) for i in list(ind)]
    return metrics
