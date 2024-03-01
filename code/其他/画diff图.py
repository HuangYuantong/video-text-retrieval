from 数据统计_画图 import draw_bar_diff

from temp.keyFrameExtractor_Uniform import keyFrameExtractor_Uniform
from temp.keyFrameExtractor_diff import keyFrameExtractor_diff

_, target_idx_u = keyFrameExtractor_Uniform(
    ('D:/Code/Python/Graduation_Project/res/MSRVTT/videos/all/video0.mp4', 0),
    'D:/Code/Python/Temp/outputs')
_, target_idx_d, idx_list, diff_list = keyFrameExtractor_diff(
    ('D:/Code/Python/Graduation_Project/res/MSRVTT/videos/all/video0.mp4', 0),
    'D:/Code/Python/Temp/outputs', 6)

target_idx_u[0] = 0
draw_bar_diff(idx_list, diff_list, target_idx_d, target_idx_u, 'D:/文档/课程资料/·毕业论文/6 答辩/资料/3 关键帧选取.svg')
