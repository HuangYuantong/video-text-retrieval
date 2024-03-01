from django.shortcuts import render
from django.db.models import Q, Sum
from .models import Category, Video_Clip
from VideoSearchSystem.VideoSearchSystem import VideoSearchSystem
import random

# 计算总和：Sum是查询语句，aggregate执行语句并存到字典里
total_video = Category.objects.aggregate(total_video=Sum('video_clip_number', default=0))['total_video']

# Create your views here.
"""分析传入的参数，（使用定义的HTML模板）构造出相应HTML文件"""


def home_valoff(request):
    sentence = request.GET.get(key='q')  # 若q不存在或为空，返回`default` = None
    infos = None
    if sentence == '':  # 测试用，空白输入随机查找一句话
        sentence = Video_Clip.objects.all()[random.randint(1, 999)].__str__()
    if sentence:
        infos = VideoSearchSystem.search(sentence, top_n=50, dataset_name=['MSRVTT_test'])[0]  # [0]是因为一次只计算一句话
    return render(request, 'website/home_valoff.html', {'sentence': sentence,  # 将sentence传回去填在搜索框里
                                                        'infos': infos,  # 该sentence对应查询结果
                                                        'mark_self': 'off',  # 标记自身的文字匹配模式是否开启
                                                        'mark_other': 'on',
                                                        'total_video': total_video,  # 视频总数量
                                                        })


def home_valon(request):
    sentence = request.GET.get(key='q')  # 若q不存在或为空，返回`default` = None
    if sentence == '':  # 测试用，空白输入随机查找一句话
        sentence = Video_Clip.objects.all()[random.randint(1, 999)].__str__()
    if not sentence:
        infos = None
        sidebar_infos = Category.objects.all()
    else:
        infos = VideoSearchSystem.search(sentence, top_n=50, dataset_name=['MSRVTT_test'])[0]  # [0]是因为一次只计算一句话
        infos = [[i[0], '', i[2]] for i in infos]  # 如果要显示视频检索模块返回的sentence，注释此句
        sidebar_infos = Video_Clip.objects.filter(Q(sentences__icontains=sentence))
    return render(request, 'website/home_valon.html', {'sentence': sentence,  # 将sentence传回去填在搜索框里
                                                       'infos': infos,  # 该sentence对应查询结果
                                                       'sidebar_infos': sidebar_infos,  # 在数据库里用文本查询的结果
                                                       'mark_self': 'on',  # 标记自身的文字匹配模式是否开启
                                                       'mark_other': 'off',
                                                       'total_video': total_video,  # 视频总数量
                                                       })


def video_player_valoff(request, pk):
    return render(request, 'website/video_player.html', {'search_in_database': False,  # 是否开启文字匹配
                                                         'video_id': pk,
                                                         'mark_self': 'off',  # 标记自身的文字匹配模式是否开启
                                                         'mark_other': 'on',
                                                         'total_video': total_video,  # 视频总数量
                                                         })


def video_player_valon(request, pk):
    video_obj = Video_Clip.objects.get(video_id=pk)
    return render(request, 'website/video_player.html', {'search_in_database': True,  # 是否开启文字匹配
                                                         'video_id': pk,
                                                         'category': video_obj.category,
                                                         'sentences': video_obj.sentences.split('\n'),
                                                         'mark_self': 'on',  # 标记自身的文字匹配模式是否开启
                                                         'mark_other': 'off',
                                                         'total_video': total_video,  # 视频总数量
                                                         })


def db_construct(request):
    from django.shortcuts import HttpResponse
    from .db_build import db_build_category, db_build_video_clip, db_build_video_clip_number
    info = '无DataBase构造任务，请勿访问。\n若要进行数据库构造，请打开./website/views/db_construct函数，参考注释进行修改'

    # info = db_build_category()  # 取消此行注释：初始化Category
    # info = db_build_video_clip()  # 取消此行注释：初始化Video_Clip
    # info = db_build_video_clip_number()  # 取消此行注释：更新各Category的video_clip_number参数

    return HttpResponse(info.replace('\n', '<br>').replace(' ', '&nbsp;'),  # 在网页上，’<br>‘显示换行’\n‘，’&nbsp;‘显示空格’ ‘
                        content_type='text/html')  # 且要指定文本为html才能解析
