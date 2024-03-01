import re
import numpy

from 数据统计_画图 import draw_line, draw_barh_single


def get_score(content: str, score_text: list):
    """从content中解析出所有score_list中的指标数值\n
    :return: numpy.ndarray[指标数量, epoch]"""
    results = [list(map(lambda x: float(x.strip()),
                        re.findall(re.compile(i + r'(\d+\.\d+)'), content)))
               for i in score_text]
    results = numpy.array(results)
    return results


def statistic(file_path: str, score_index: list[str], score_text: list[str],
              draw: bool = True, save_path: str = None,
              xlabel: str = '', ylabel: str = '', xticks=None):
    """
    :param file_path: log保存路径
    :param score_index: 寻找的指标（画图时图例用命）
    :param score_text: 指标在文件中的字符

    :param draw: True：进行画图，False：print输出，后面3个参数失效
    :param save_path: 画图保存路径，为None时只显示不保存
    :param xlabel: 画图的横坐标标签
    :param ylabel: 画图的纵坐标标签
    :param xticks: 控制x坐标轴上显示的刻度，为None则matplotlib自适应
    """
    with open(file_path, 'r', encoding='utf-8') as file:  # 读入带分割数据
        content = file.read()
    results = get_score(content, score_text)  # 构造所有数据
    if draw:
        draw_line(score_index, results, save_path, xlabel, ylabel, xticks)  # 画图
    else:
        print(results)
        print('形状：', results.shape)
        print('统计目标：', score_index)
    return results


if __name__ == '__main__':
    # 画横着的柱状图
    draw_barh_single(['CLIP4Clip', 'CLIP4Clip\n+帧保存', 'AIM\n+帧保存'], [42.58, 3.38, 1.42], '1.svg', '方案', '训练用时/h')

    # 统计训练step平均时间
    # _results = statistic('inputs.txt', ['time'], ['Time/step: '],
    #                      draw=False, save_path=None, xlabel='', ylabel='')
    # _results = _results[0]
    # print(f'统计结果：max: {_results.max():.2f}, min: {_results.min():.2f}, median: {numpy.median(_results):.2f}, '
    #       f'mean: {(_results.sum() - _results.max() - _results.min()) / (len(_results) - 2):.2f}（去除1个最高和最低）')

    # 画loss图
    # statistic('inputs.txt', ['loss'], [', Loss: '], True, None, '', 'loss', [])

    # 画2个loss一起的图
    # _result1 = statistic('inputs1.txt', ['loss'], [', Loss: '], False, None, '', 'loss', [])
    # _result2 = statistic('inputs2.txt', ['loss'], [', Loss: '], False, None, '', 'loss', [])
    # draw_line(['原始', '帧保存'], [_result1[0], _result2[0]], None, '', 'loss', [i*56 for i in [1,2,3,4,5]])
