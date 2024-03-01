def get_info(shuffle: bool = True):
    import json
    import numpy
    import pandas
    import os
    from collections import defaultdict
    from Tools import name2index
    from get_args import DIR_RES, dir_MSRVTT_data_train_9k, dir_MSRVTT_data

    # 句子
    data = json.load(open(dir_MSRVTT_data, 'r'))  # 'info', 'videos', 'sentences'
    # defaultdict会自动初始化第一个值，只需关注添加元素
    # 保留元素插入的顺序用列表，希望消除重复元素（并且不在意他们的排序）用集合
    video_id_sentence_dict = defaultdict(list)  # {video_id:[sentence]}
    for item in data['sentences']:
        video_id_sentence_dict[name2index(item['video_id'])].append(item['caption'])  # {video_id:[sentence]}
    # video_id
    df_msrvtt_train = pandas.read_csv(dir_MSRVTT_data_train_9k)

    # 打乱顺序
    if shuffle:
        batches = numpy.array(range(20))
        numpy.random.shuffle(batches)
        # 构建
        train_data = list()
        for batch in batches:
            video_ids = list(map(name2index, df_msrvtt_train.sample(frac=1)['video_id']))  # sample随机打乱顺序
            sentences = list(map(lambda x: video_id_sentence_dict[x][batch], video_ids))  # video_id对应的第batch个句子
            train_data.extend(zip(video_ids, sentences))
        # 保存
        pandas.DataFrame(columns=['video_id', 'sentence'], data=train_data
                         ).to_csv(os.path.join(DIR_RES, 'train.csv'), index=False)
    # 顺序保存
    else:
        # 构建
        train_data = list()
        for video_id in list(map(name2index, df_msrvtt_train['video_id'])):
            for sentence in video_id_sentence_dict[video_id]:
                train_data.append(tuple([video_id, sentence]))
        # 保存
        pandas.DataFrame(columns=['video_id', 'sentence'], data=train_data
                         ).to_csv(os.path.join(DIR_RES, 'train_order.csv'), index=False)
