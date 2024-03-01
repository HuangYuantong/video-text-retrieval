import os
from concurrent import futures
from functools import partial
from time import time
from typing import Union

import cv2
import numpy
import pandas
from PIL import Image

from Tools import name2index


def extract_all_videos(frameExtractor, inputs: Union[str, list[str]], output_path: str, key_frames_data_path: str):
    """抽取文件夹下所有视频的关键帧、保存\n
    :param frameExtractor: 使用的帧抽取函数
    :param inputs: 要抽取的文件夹（不会递归遍历）、或要抽取的所有视频路径
    :param output_path: 抽取结果保存路径（结果命名：{video_id}_{idx}.jpg）
    :param key_frames_data_path: 关键帧数量信息文件路径
    """
    if isinstance(inputs, str):
        video_names = list(i for i in os.listdir(inputs) if i.endswith('.mp4'))  # input_path文件夹下所有视频文件名
        video_paths = map(lambda i: os.path.join(inputs, i), video_names)
        video_ids = list(map(name2index, video_names))  # 所有文件名通过name2index映射为video_id
    else:
        video_paths = inputs
        video_ids = list(map(lambda x: name2index(x.rsplit(os.sep, 1)[-1]), video_paths))
    # 指定extract_video的输入路径、输出路径
    if not os.path.exists(output_path): os.mkdir(output_path)
    _frameExtractor = partial(frameExtractor, output_path=output_path)
    # 遍历处理所有视频，抽取保存关键帧、返回各视频关键帧数量
    time1 = time()
    # 多线程：
    # with Pool(12) as p:
    #     # map异步、结果顺序、返回list，imap异步、结果顺序、返回iter，都是对map_async的包装
    #     # map传入iterable在实际使用中，必须整个队列都就绪后才会运行子进程 ==> map比imap快但占空间
    #     key_frame_counts = p.map(_frameExtractor, list(zip(video_paths, video_ids)))
    with futures.ThreadPoolExecutor() as thread:  # max_workers = min(32, (os.cpu_count() or 1) + 4)
        key_frame_counts = thread.map(_frameExtractor, list(zip(video_paths, video_ids)))  # chunksize参数只对进程池有效
    key_frame_counts = list(key_frame_counts)
    # 多进程
    # with futures.ProcessPoolExecutor() as process:  # 进程数默认为CPU虚拟内核数
    #     key_frame_counts = process.map(_frameExtractor, list(zip(video_paths, video_ids)), chunksize=125)
    # key_frame_counts = list(key_frame_counts)
    # 循环
    # key_frame_counts = numpy.zeros(len(video_ids), dtype=int)
    # for idx, video_path_id in tqdm(enumerate(zip(video_paths, video_ids)), total=len(video_ids), mininterval=5):
    #     key_frame_counts[idx] = _frameExtractor(video_path_id=video_path_id)
    time2 = time()
    print(f'耗时：{time2 - time1:.4f}s，平均：{(time2 - time1) / len(video_ids):.4f}s')
    _update_key_frames_data(video_ids, key_frame_counts, key_frames_data_path)


def _update_key_frames_data(video_ids: list[int], key_frame_counts: Union[list[int], numpy.ndarray], key_frames_data_path: str):
    """更新 key_frames_data.csv\n
    :param video_ids: 各视频对应的video_id
    :param key_frame_counts: 各视频对应的关键帧数量
    :param key_frames_data_path: 关键帧数量信息文件路径
    """
    # 保存video_id到key_frame_count的映射到文件
    df = pandas.DataFrame(columns=['video_id', 'key_frame_count'], data=zip(video_ids, key_frame_counts))
    # 文件存在的话就合并
    if os.path.exists(key_frames_data_path):
        df = pandas.concat([df, pandas.read_csv(key_frames_data_path)]
                           ).drop_duplicates(subset=['video_id'], keep='first')  # 交集部分保留本次新数据
    df.sort_values(by='video_id').to_csv(key_frames_data_path, index=False)
    # 统计一下关键帧数量情况
    info = dict((i, key_frame_counts.count(i) if isinstance(key_frame_counts, list) else numpy.sum(key_frame_counts == i))
                for i in range(max(key_frame_counts) + 1))
    print([f'帧数量{i}: {count}, {count / len(key_frame_counts) * 100}%' for i, count in info.items()])
    pandas.DataFrame(columns=['frame_number', 'count'], data=list((i, count) for i, count in info.items())
                     ).to_csv(os.path.join(os.path.dirname(key_frames_data_path), '关键帧数量统计.csv'), mode='a', index=False)


#############################################
# 图片resize、中心裁剪

def resize_centralCrop(image: Union[Image.Image, numpy.ndarray], size):
    """PIL.Image图像“最小边缩放至size、再中心裁剪；
    size<=0时，不缩放只中心裁剪。
    """
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    if height < width:
        if size > 0: width, height = int(width * (size / height)), size
        left, upper, right, lower = (width - height) // 2, 0, (width - height) // 2 + height, height
    else:
        if size > 0: width, height = size, int(height * (size / width))
        left, upper, right, lower = 0, (height - width) // 2, width, (height - width) // 2 + width

    if isinstance(image, Image.Image):
        return image.resize((width, height), Image.Resampling.BICUBIC  # 最小边缩放至size
                            ).crop((left, upper, right, lower))  # 中心裁剪
    else:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR  # 最小边缩放至size
                          )[upper:lower, left:right]  # 裁剪


def get_resize_centralCrop_size(image: Union[Image.Image, numpy.ndarray, tuple], size):
    """
    计算图片（最小边缩放到size）的：height, width（size<=0时，不缩放）；\n
    以及要进行中心裁剪的：left, upper, right, lower；\n
    如果要直接传入图片高宽，注意次序：image=(height, width)；\n
    :return: height, width, left, upper, right, lower
    """
    if isinstance(image, Image.Image):
        width, height = image.size
    elif isinstance(image, numpy.ndarray):
        height, width = image.shape[:2]
    else:
        height, width = int(image[0]), int(image[1])
    if height < width:
        if size > 0: width, height = int(width * (size / height)), size
        left, upper, right, lower = (width - height) // 2, 0, (width - height) // 2 + height, height
    else:
        if size > 0: width, height = size, int(height * (size / width))
        left, upper, right, lower = 0, (height - width) // 2, width, (height - width) // 2 + width
    return height, width, left, upper, right, lower


def execute_resize_centralCrop(image: Union[Image.Image, numpy.ndarray], height=-1, width=-1,
                               left=-1, upper=-1, right=-1, lower=-1) -> Union[Image.Image, numpy.ndarray]:
    """PIL.Image图像最小边缩放至height, width（三线性差值）、再裁剪；
    height, width 任一<=0时，不缩放；
    left, upper, right, lower 任一<0时不裁剪。
    """
    _image = image
    if isinstance(_image, Image.Image):
        if height > 0 and width > 0:
            _image = _image.resize((width, height), Image.Resampling.BICUBIC)  # 最小边缩放至size
        if left >= 0 and upper >= 0 and right >= 0 and lower >= 0:
            _image = _image.crop((left, upper, right, lower))  # 裁剪
    else:
        if height > 0 and width > 0:
            _image = cv2.resize(_image, (width, height), interpolation=cv2.INTER_LINEAR)  # 最小边缩放至size
        if left >= 0 and upper >= 0 and right >= 0 and lower >= 0:
            _image = _image[upper:lower, left:right]  # 裁剪
    return _image
