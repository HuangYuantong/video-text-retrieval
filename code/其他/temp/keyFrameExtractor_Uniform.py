import os
import cv2

from .frame_extract_tool import get_resize_centralCrop_size, execute_resize_centralCrop


def keyFrameExtractor_Uniform(video_path_id: tuple[str, int], output_path: str,
                              max_frame=12):
    """视频不超过 max_frame 秒时，取每秒的第1帧；\n
    超过 max_frame 秒时，平均取 max_frame 帧；\n
    :return: 该视频中抽取出的关键帧数量
    """
    video_path, video_id = video_path_id
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # height, width, left, upper, right, lower
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size_crop_info = get_resize_centralCrop_size((height, width), 224)

    # 得到目标帧的idx
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    # <== 使用 ‘(total_frame + fps - 1) // fps’ 以和CLIP4Clip原代码保持一致
    # 视频不超过12秒，每1秒取第1帧
    if (total_frame + fps - 1) // fps <= max_frame:  # <== 使用 ‘(total_frame + fps - 1) // fps’ 以和CLIP4Clip原代码保持一致
        target_idx = list(range(total_frame))[::fps]
    else:  # 超过12秒的视频，平均间隔取12帧
        target_idx = [i * (total_frame // max_frame) for i in range(max_frame)]
    target_idx[0] = 2  # MSRVTT数据集部分有问题，第1帧内容不符合片段内容
    # 取出目标帧并保存
    _length = 0
    for idx in target_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break  # <== 使用 ‘break’ 而不是重新尝试一次open 以和CLIP4Clip原代码保持一致
        _length += 1
        cv2.imwrite(os.path.join(output_path, f'{video_id}_{_length - 1}.jpg'),
                    execute_resize_centralCrop(frame, *size_crop_info),  # 缩放并中心裁剪
                    [cv2.IMWRITE_JPEG_QUALITY, 95])  # jpg有损压缩：0~100（100文件最大，压缩最低但仍有），默认95
    cap.release()
    return _length, target_idx
