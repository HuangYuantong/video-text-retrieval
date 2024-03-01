def db_build_category():
    from .models import Category

    new = [Category(category=0, name='music', video_clip_number=1064),
           Category(category=1, name='people', video_clip_number=338),
           Category(category=2, name='gaming', video_clip_number=639),
           Category(category=3, name='sports', video_clip_number=857),
           Category(category=4, name='news', video_clip_number=467),
           Category(category=5, name='education', video_clip_number=270),
           Category(category=6, name='tv shows', video_clip_number=330),
           Category(category=7, name='movie', video_clip_number=1168),
           Category(category=8, name='animation', video_clip_number=458),
           Category(category=9, name='vehicles', video_clip_number=599),
           Category(category=10, name='howto', video_clip_number=803),
           Category(category=11, name='travel', video_clip_number=148),
           Category(category=12, name='science', video_clip_number=293),
           Category(category=13, name='animals', video_clip_number=428),
           Category(category=14, name='kids', video_clip_number=520),
           Category(category=15, name='doc', video_clip_number=148),
           Category(category=16, name='food', video_clip_number=504),
           Category(category=17, name='cooking', video_clip_number=356),
           Category(category=18, name='beauty', video_clip_number=377),
           Category(category=19, name='ads', video_clip_number=233), ]
    Category.objects.bulk_create(new)
    # 构造消息用于显示
    info = list(f' ==> Category构建完毕\n共添加{len(new)}条Category')
    return '\n'.join(info)


def db_build_video_clip():
    import os
    import json
    import pandas
    import numpy
    from collections import defaultdict
    from .models import Category, Video_Clip

    from VideoSearchSystem.get_args import dir_MSRVTT_data, DIR_DataBase
    from VideoSearchSystem.Tools import name2index

    data = json.load(open(dir_MSRVTT_data, 'r'))  # 'info', 'videos', 'sentences'
    # defaultdict会自动初始化第一个值，只需关注添加元素
    # 保留元素插入的顺序用列表，希望消除重复元素（并且不在意他们的排序）用集合
    video_id_sentence_dict = defaultdict(list)
    video_id_category_dict = dict()
    for item in data['sentences']:
        video_id_sentence_dict[name2index(item['video_id'])].append(item['caption'])  # {video_id:[sentence]}
    for item in data['videos']:
        video_id_category_dict[name2index(item['video_id'])] = item['category']  # {video_id:category}
    print('video_id_sentence_dict、video_id_category_dict构建完毕')
    # 只将video_id_to_sentence.csv文件中的视频添加到数据库中
    video_id_list = numpy.array(pandas.read_csv(os.path.join(DIR_DataBase, 'video_id_to_sentence.csv'))['video_id'])
    # 批量创建
    Video_Clip.objects.bulk_create(
        list(Video_Clip(video_id=video_id,
                        category=Category.objects.get(category=video_id_category_dict[video_id]),
                        sentences='\n'.join(video_id_sentence_dict[video_id]))  # 多个sentence间用一次换行分割
             for video_id in video_id_list)  # 只构建test集。构建所有：in video_id_sentence_dict.keys()
    )
    # 构造消息用于显示
    info = list()
    info.append(f' ==> Video_Clip构建完毕\n共添加{len(video_id_list)}条Video_Clip')
    info.append(db_build_video_clip_number())  # 更新一下Category.video_clip_number
    return '\n'.join(info)


def db_build_video_clip_number():
    from django.db.models import Count
    from .models import Category

    new = Category.objects.annotate(temp_num=Count('video_clip'))  # Count可以直接查询连接的外键数，annotate将数据暂存在新创建的属性中
    Category.objects.bulk_update(  # 批量更新
        objs=list(Category(category=i.category, video_clip_number=i.temp_num) for i in new),
        fields=['video_clip_number'])
    # 构造消息用于显示
    info = list(' ==> 更新Category.video_clip_number')
    info.append(f'{"category":<10},{"name":<20}, {"video_clip_number":<10} -> {"temp_num":<10}')
    info.extend([f'{i.category:<10}, {i.name:<20}, {i.video_clip_number:<10} -> {i.temp_num:<10}' for i in new])
    info.append(f'sum(temp_num) = {sum(i.temp_num for i in new)}')
    return '\n'.join(info)
