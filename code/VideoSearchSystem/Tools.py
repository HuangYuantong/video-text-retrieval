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


def evaluation(model, device):
    """在test集上计算相似矩阵的R1、R5、R10、MedianR、MeanR"""
    from TensorDataBase.TensorDataBase import TensorDataBase
    with torch.no_grad():
        _, text_feature, video_feature = TensorDataBase.load(TensorDataBase.MSRVTT_test)
        text_feature, video_feature = torch.from_numpy(text_feature).to(device, torch.float32), \
                                      torch.from_numpy(video_feature).to(device, torch.float32)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)  # L2正则
        video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)

        sim_matrix = model.get_similarity_logits(text_feature, video_feature).cpu().numpy()
        return show_score(sim_matrix)


def evaluation_max(model, device):
    """取12帧里相似度最大值作为视频相似度\n
    在test集上计算相似矩阵的R1、R5、R10、MedianR、MeanR"""
    from TensorDataBase.TensorDataBase import TensorDataBase
    with torch.no_grad():
        _, text_feature, video_feature = TensorDataBase.load(TensorDataBase.MSRVTT_test_raw)
        text_feature, video_feature = torch.from_numpy(text_feature).to(device, torch.float32), \
                                      torch.from_numpy(video_feature).to(device, torch.float32)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)  # L2正则
        video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)

        b, f, d = video_feature.shape  # [N, max_frame, 512]
        video_feature = video_feature.view(b * f, d)  # [N*max_frame, 512]
        sim_matrix = model.get_similarity_logits(text_feature, video_feature)  # [N, N*max_frame]
        sim_matrix = sim_matrix.view(b, b, f)  # [N, N, max_frame]
        sim_matrix = torch.max(sim_matrix, dim=-1, keepdim=False)[0].cpu().numpy()  # [N, N]
        return show_score(sim_matrix)


def show_score(sim_matrix):
    """输出Text-to-Video、Video-to-Text得分"""
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MedianR'], tv_metrics['MeanR']))
    print("Video-to-Text:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MedianR'], vt_metrics['MeanR']))
    return tv_metrics


#############################################
# 其他

def split_run_extract_all_videos(inputs: Union[str, list[str]], uniform=True, diff=False):
    """将视频数据量划分为1000一份，分批次调用 extract_all_videos；
    若uniform为True：进行keyFrameExtractor_Uniform；
    若diff为True，进行keyFrameExtractor_diff"""
    from preprocess.frame_extract.frame_extract_tool import extract_all_videos
    from preprocess.frame_extract.keyFrameExtractor_Uniform import keyFrameExtractor_Uniform
    from preprocess.frame_extract.keyFrameExtractor_diff import keyFrameExtractor_diff
    from .get_args import dir_key_frames_uniform, dir_key_frames_diff

    if isinstance(inputs, str):
        all_video_paths = map(lambda i: os.path.join(inputs, i),
                              list(i for i in os.listdir(inputs) if i.endswith('.mp4')))  # input_path文件夹下所有视频文件名
    else:
        all_video_paths = inputs
    # 太多了，分成1000个视频一份以防意外
    n = len(all_video_paths) // 1000 if len(all_video_paths) % 1000 == 0 \
        else len(all_video_paths) // 1000 + 1  # 向上取整
    # 平均取帧
    if uniform:
        frameExtractor = keyFrameExtractor_Uniform
        for i in range(n):
            video_paths = all_video_paths[1000 * i:1000 * (i + 1)]
            extract_all_videos(frameExtractor, video_paths, dir_key_frames_uniform,
                               os.path.join(dir_key_frames_uniform, 'key_frames_data.csv'))
            print(f'keyFrameExtractor_Uniform：{len(video_paths)}个视频完成：i={i}已结束，下次i从{i + 1}（n={n}）开始')
    if diff:
        # 帧间差
        frameExtractor = keyFrameExtractor_diff
        for i in range(n):
            video_paths = all_video_paths[1000 * i:1000 * (i + 1)]
            extract_all_videos(frameExtractor, video_paths, dir_key_frames_diff,
                               os.path.join(dir_key_frames_diff, 'key_frames_data.csv'))
            print(f'keyFrameExtractor_diff：{len(video_paths)}个视频完成：i={i}已结束，下次i从{i + 1}（n={n}）开始')


def video_to_thumbnail(video_path: str, output_path: str, height=180, width=320, jpg_quality=95):
    """将文件夹下所有视频取第5帧作为封面；\n
    保存jpg在store_path中（180:320=9:16），图片名称使用DataBase.name2index映射结果（int.jpg）"""
    import cv2
    from tqdm import tqdm
    if not os.path.exists(video_path): exit('video_to_thumbnail函数：输入路径不存在')
    if not os.path.exists(output_path): os.mkdir(output_path)
    for file_name in tqdm(os.listdir(video_path), mininterval=1):
        if file_name.startswith('.') or (not file_name.endswith(('.mp4',))): continue
        # 取第5帧作为封面
        cap = cv2.VideoCapture(os.path.join(video_path, file_name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
        _, frame = cap.read()
        # 统一尺寸为 width:height（9:16)
        h, w, _ = frame.shape
        if not (h == height and w == width):
            frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC三线性插值，耗时最长
        # 保存为jpg
        cv2.imwrite(os.path.join(
            output_path,
            f'{name2index(file_name.rsplit(".", maxsplit=1)[0])}.jpg'),  # 图片名称使用DataBase.name2index映射结果
            frame,  # 统一尺寸为 width:height（9:16)
            [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])  # jpg有损压缩：0~100（100文件最大，压缩最低但仍有），默认95
        cap.release()
