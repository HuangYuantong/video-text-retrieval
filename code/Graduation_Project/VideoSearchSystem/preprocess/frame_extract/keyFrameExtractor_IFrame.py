import os
import av

from .frame_extract_tool import resize_centralCrop


def keyFrameExtractor_IFrame(video_path: str, video_id: int, output_path: str) -> int:
    """用FFmpeg抽取I帧（不好，I帧受视频格式等无关因素影响）\n
    :return: 该视频中抽取出的关键帧数量
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'  # NONKEY：跳过关键帧
    idx = 0
    for frame in container.decode(stream):
        # 小边缩放至224、再中心裁剪
        resize_centralCrop(frame.to_image(), 224  # to_image()转为< PIL.Image.Image image mode=RGB >
                           ).save(os.path.join(output_path, f'{video_id}_{idx}.jpg'), quality=100)  # 保存为jpg
        idx += 1
    return idx
