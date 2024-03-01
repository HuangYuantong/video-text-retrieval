import matplotlib.pylab as plt
from matplotlib import font_manager
import numpy
import pandas

file_name = 'AIM得分'
epoch = 40
score_index = ['R@1', 'R@5', 'R@10', 'MdR', 'MnR']  # 寻找的指标
score_text = ['R@1: ', 'R@5: ', 'R@10: ', 'Median R: ', 'Mean R: ']  # 指标在文件中的字符


def get_score(content, score_text):
    """从content中解析出所有score_list中的指标数值\n
    :return: numpy.ndarray[5, epoch]"""
    start = [content[0].index(i) + len(i) for i in score_text]  # 各指标数值开始位置
    score = numpy.array([[float(line[i:i + 6].strip(' -')) for i in start]  # 取出各指标、转为float
                         for line in content]  # 遍历所有epoch
                        ).transpose()  # 转置：[epoch, 5]->[5, epoch]
    return score


def to_pandas():
    """存储到表格"""
    df = pandas.DataFrame(columns=['R@1', 'R@5', 'R@10', 'MdR', 'MnR', 'R@1', 'R@5', 'R@10', 'MdR', 'MnR'],
                          data=numpy.hstack([score_mean.transpose(), score_max.transpose()]))
    df['epoch'] = epoch_list
    df = df.set_index('epoch')
    df.to_csv(f'./output/{file_name}.csv')


def to_matplotlib():
    """画图输出"""
    # 设置折线
    plt.figure(figsize=(4, 3))  # 画布大小
    # color, linewidth,marker, markersize, markeredgewidth, markerfacecolor, markeredgecolor
    # R5
    plt.plot(epoch_list, score_mean[1], label=f'meanP: {score_index[1]}',
             color='dodgerblue', linewidth=1, marker='s', markersize=3, markeredgewidth=0)
    plt.plot(epoch_list, score_max[1], label=f' max : {score_index[1]}',
             color='orangered', linewidth=1, marker='o', markersize=3, markeredgewidth=0)
    # R1
    plt.plot(epoch_list, score_mean[0], label=f'meanP: {score_index[0]}',
             color='steelblue', linewidth=1, marker='s', markersize=3, markeredgewidth=0)
    plt.plot(epoch_list, score_max[0], label=f' max : {score_index[0]}',
             color='darkred', linewidth=1, marker='o', markersize=3, markeredgewidth=0)
    # 设置x轴label、旋转角度、标题等
    font_title = font_manager.FontProperties(family='Times New Roman', size=11)
    font_index = font_manager.FontProperties(family='Times New Roman', size=8)
    plt.xlabel('epoch', fontproperties=font_title)
    plt.ylabel('recall', fontproperties=font_title)
    plt.xticks(fontproperties=font_index)  # 设置字体
    plt.yticks(fontproperties=font_index)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    plt.legend(loc='best')  # 显示图例说明
    plt.tight_layout()  # 减小四周空白距离
    # plt.savefig(f'./output/{file_name}.svg')  # 保存
    plt.show()  # 输出


if __name__ == '__main__':
    with open(f'./output/{file_name}.txt', 'r', encoding='utf-8') as file:  # 读入带分割数据
        _content = file.readlines()
        content_mean = _content[2::11]
        content_max = _content[6::11]

    # 构造所有数据
    epoch_list = numpy.array(range(epoch + 1), int)  # epoch：[epoch, ]
    score_mean = numpy.hstack(([[i] for i in [30.9, 53.6, 63.3, 4.0, 41.6]], get_score(content_mean, score_text)))
    score_max = numpy.hstack(([[i] for i in [32.102, 55.427, 66.628, 4.0, 31.591]], get_score(content_max, score_text)))
    # to_pandas()
    to_matplotlib()
