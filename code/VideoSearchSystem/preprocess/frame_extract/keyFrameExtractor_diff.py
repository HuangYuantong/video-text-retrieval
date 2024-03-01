import operator
import os

import cv2
import numpy
from scipy import signal

from .frame_extract_tool import get_resize_centralCrop_size, execute_resize_centralCrop


class _Frame_Info:
    def __init__(self, idx, diff):
        self.idx = idx  # 第id帧
        self.diff = diff  # 与上一帧的帧间差距


def keyFrameExtractor_diff(video_path_id: tuple[str, int], output_path: str,
                           frame_step=6, max_frame=12, method='to_k') -> int:
    """用cv2依据帧间差选取关键帧\n
    :return: 该视频中抽取出的关键帧数量
    """
    video_path, video_id = video_path_id
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # height, width, left, upper, right, lower
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size_crop_info = get_resize_centralCrop_size((height, width), 224)

    # 第1遍遍历视频，计算所有diff
    frames = _get_diff(cap, frame_step)
    # 得到目标帧的idx
    if method == 'local_maxima':
        target_idx = _diff_local_maxima(frames, int(cap.get(cv2.CAP_PROP_FPS) / frame_step), 'hamming')
    else:
        target_idx = _diff_top_k(frames, max_frame)
    del frames
    # 第2遍遍历视频，取出目标帧并保存
    for _i, idx in enumerate(target_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # print(f'结果有{idx} ({int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})')
        _, frame = cap.read()
        if frame is None:
            cap.open(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            print(f'{video_path}:{idx} ({int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})出错，'
                  f'已重新open试图恢复')
            _, frame = cap.read()
        cv2.imwrite(os.path.join(output_path, f'{video_id}_{_i}.jpg'),
                    execute_resize_centralCrop(frame, *size_crop_info),  # 缩放并中心裁剪
                    [cv2.IMWRITE_JPEG_QUALITY, 95])  # jpg有损压缩：0~100（100文件最大，压缩最低但仍有），默认95
    cap.release()
    return len(target_idx)


def _get_diff(cap: cv2.VideoCapture, frame_step: int) -> list[_Frame_Info]:
    """第1遍遍历视频，计算所有diff"""
    # height, width, left, upper, right, lower
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size_crop_info = get_resize_centralCrop_size((height, width), -1)  # 不缩放，只中中心裁剪
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    frames = list()  # 存储所有[(idx, diff)]
    # 跳过第1帧，MSRVTT数据集部分有问题，第1帧内容不符合片段内容
    ret = cap.grab()  # grab跳帧，不解析图片所以比read更快
    idx = 0  # 当前第idx帧（初始值为-1）
    pre_frame = None  # 前一帧
    while ret:
        idx += 1  # 当前第idx帧
        if idx + 2 == total_frame: break  # 若idx==total_frame被grab，那么后续cap无法再跳转、解析。这里防止意外设大一点
        if (idx - 1) % frame_step != 0:  # 设置每 k 个取一次帧
            ret = cap.grab()
            continue
        # print(f'抓取到{idx} ({int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})')
        ret, frame = cap.read()
        if not ret:  # 仍然可能由于视频或cv2问题出现解析失败
            print(f'_get_diff中：{idx} ({int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})出错，'
                  f'已提前结束第一次循环试图继续运行')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度，加快计算
        frame = execute_resize_centralCrop(frame, -1, -1, *size_crop_info[2:])  # 中心裁剪，调整为网络能看到的视野
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        if pre_frame is not None:
            diff = cv2.absdiff(frame, pre_frame)  # 计算帧间差距（绝对值）
            diff = cv2.medianBlur(diff, 5)  # medianBlur适用于去除孤立点（椒盐噪声）
            frames.append(_Frame_Info(idx, numpy.sum(diff)))  # 帧间差距总和
        pre_frame = frame
    del pre_frame
    return frames


def _diff_top_k(frames: list[_Frame_Info], k: int):
    """取diff最大的前k个，返回[idx]（升序）"""
    frames.sort(key=operator.attrgetter('diff'), reverse=True)
    return sorted([i.idx for i in frames[:k]])


def _diff_local_maxima(frames: list[_Frame_Info], window_len=13, window='hamming'):
    """使用Local Maxima，返回[idx]（升序）"""
    diff_array = smooth(numpy.array(list([i.diff for i in frames]), int),
                        window_len, window)  # 平滑
    frame_idx = signal.argrelextrema(diff_array, numpy.greater)[0]  # 取极大点
    return list([frames[i].idx for i in frame_idx])


def _diff_thresh(frames: list[_Frame_Info], thresh: int):
    """使用阈值分割，返回[idx]（升序）"""

    def rel_change(a, b):
        x = (b - a) / max(a, b)
        return x

    result = list()
    for i in range(1, len(frames)):
        if rel_change(frames[i - 1].diff, frames[i].diff) >= thresh:
            result.append(frames[i].idx)
    return result


def smooth(x, window_len=13, window='hamming'):
    """数据平滑"""
    # window有：'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    # 镜面padding。numpy.r_[] 等价 numpy.hstack()
    s = numpy.hstack([x[(window_len // 2):0:-1], x,
                      x[-2:-(window_len // 2) - 2:-1]])
    if window == 'flat':  # 均值窗口
        w = numpy.ones(window_len)
    else:
        w = getattr(numpy, window)(window_len)
    y = numpy.convolve(s, w / w.sum(), mode='valid')
    return y
