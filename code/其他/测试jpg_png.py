import os
import cv2


def _png_jpg_test(image):
    for i in range(10):
        cv2.imwrite(os.path.join(f'png-{i}.png'),  # 图片名称使用DataBase.name2index映射结果
                    image, [cv2.IMWRITE_PNG_COMPRESSION, i])  # png无损压缩：0~9（9文件最小，解码最慢），默认3
    for i in range(21):
        cv2.imwrite(os.path.join(f'jpg-{i * 5}.jpg'),  # 图片名称使用DataBase.name2index映射结果
                    image, [cv2.IMWRITE_JPEG_QUALITY, i * 5])  # jpg有损压缩：0~100（100文件最大，压缩最低但仍有），默认95
