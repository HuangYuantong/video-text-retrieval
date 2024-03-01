import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
import numpy
import re
from typing import Union

font_ch = FontProperties(family='SimSun', size=10.5)  # 创建中文字体对象（宋体）
font_en = FontProperties(family='Times New Roman', size=10.5)  # 创建英文字体对象（Times New Roman）


def match_font(s: Union[str, list[str]]):
    """判断str或list[str]中是否包含中文字符，从而返回对应字体"""
    pattern = re.compile(r'[\u4E00-\u9FEF]')  # 匹配中文字符的正则表达式
    result = pattern.search(s) if isinstance(s, str) else \
        any(pattern.search(i) for i in s)
    return font_ch if result else font_en  # 在字符串中搜索匹配项


def draw_line(score_index, results, save_path, xlabel: str, ylabel: str, xticks):
    """画折线图"""
    plt.figure(figsize=(4, 3))  # 画布大小
    epoch_list = numpy.array(range(len(results[0])), int)  # epoch：[epoch, ]
    # 设置折线
    # color, linewidth,marker, markersize, markeredgewidth, markerfacecolor, markeredgecolor
    if len(score_index) == 1:
        plt.plot(epoch_list, results[0], label=f'{score_index[0]}', color='steelblue', linewidth=1)
    else:
        for i in range(len(score_index)):
            plt.plot(epoch_list, results[i], label=f'{score_index[i]}', linewidth=1)
    # 设置label、旋转角度、标题等
    plt.xlabel(xlabel, fontproperties=match_font(xlabel))  # 坐标轴标签
    plt.ylabel(ylabel, fontproperties=match_font(xlabel))
    plt.xticks(fontproperties=font_en) if xticks is None \
        else plt.xticks(xticks, fontproperties=font_en)  # 坐标轴上坐标的标签
    plt.yticks(fontproperties=font_en)
    # 显示与保存
    plt.legend(loc='best', prop=match_font(score_index))  # 显示图例说明
    plt.tight_layout()  # 减小四周空白距离
    if save_path: plt.savefig(save_path)  # 保存
    plt.show()  # 显示


def draw_barh_single(score_index: list, results: list, save_path, xlabel: str, ylabel: str):
    """画横向柱状图（一项数据指标）"""
    plt.figure(figsize=(6, 2))  # 画布大小
    # 设置横向柱状图
    # y y轴坐标,width 条的长度,height 条的粗细,left 条的初始位置,align,hatch,color,lw,tick_label,edgecolor,label等
    bars = plt.barh(score_index[::-1], results[::-1], height=0.3)
    # 添加数值标签
    plt.bar_label(bars, labels=[round(i, 2) for i in results[::-1]], fontproperties=font_en, padding=4)
    # 设置label、旋转角度、标题等
    plt.ylabel(xlabel, fontproperties=match_font(ylabel), loc='top')  # 坐标轴标签
    plt.xlabel(ylabel, fontproperties=match_font(xlabel), loc='right')
    plt.xlim(0, max(results) + 5)  # 设置范围防止数值标签越过边界
    xticks_pos = range(0, int(max(results) + 5), 5)
    xticks_labels = [str(x) for x in xticks_pos]
    plt.xticks(xticks_pos, xticks_labels, fontproperties=font_en)  # 坐标轴上坐标的标签
    plt.yticks(fontproperties=font_ch)
    # 显示与保存
    plt.tight_layout()  # 减小四周空白距离
    if save_path: plt.savefig(save_path)  # 保存
    plt.show()  # 显示


def draw_bar_diff(idx_list_d: list[int], diff_list_d: list, target_idx_d: list[int], target_idx_u: list[int], save_path):
    """画帧间差的柱状图"""
    plt.figure(figsize=(6,3.6))  # 画布大小
    plt.bar(idx_list_d, diff_list_d, 5, color='gray')
    # 修改相应位置柱的颜色
    bars = plt.bar(target_idx_d, [diff_list_d[idx_list_d.index(i)] for i in target_idx_d], 5, color='steelblue')
    # 添加箭头
    points = plt.scatter(target_idx_u, [1e5] * len(target_idx_u), s=80, c='red', marker='v', edgecolors='black', linewidths=1)

    # 设置label、旋转角度、标题等
    plt.xlabel('帧', fontproperties=font_ch, loc='right')  # 坐标轴标签
    plt.ylabel('帧间差', fontproperties=font_ch, loc='top')
    plt.xticks(fontproperties=font_en)  # 坐标轴上坐标的标签
    plt.yticks(fontproperties=font_en)
    # 显示与保存
    plt.legend([points, bars], ['平均选取', '最大帧间差选取'],
               loc='best', prop=font_ch)  # 显示图例说明
    plt.tight_layout()  # 减小四周空白距离
    if save_path: plt.savefig(save_path)  # 保存
    plt.show()  # 显示
