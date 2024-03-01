
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
