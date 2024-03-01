import numpy
import torch
import pandas
import os
from typing import Union

from Model.My_Model import load_model
from Model.clip.clip import tokenize
from TensorDataBase.TensorDataBase import TensorDataBase
from Tools import set_seed
from get_args import get_args, DIR_MODEL, DIR_DataBase

args = get_args()  # 所有超参数
set_seed(args.seed)  # 设置随机数种子
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class VideoSearchSystem:
    # 模型
    _model = load_model(DIR_MODEL, device=device, max_frames=args.max_frames).eval()
    # DataBase中保存的特征均已在construct函数中经过L2正则
    # MSRVTT_test集数据
    _video_id_MSRVTT_test, _text_feature_MSRVTT_test, _video_feature_MSRVTT_test = TensorDataBase.load(TensorDataBase.MSRVTT_test)
    _text_feature_MSRVTT_test = torch.tensor(_text_feature_MSRVTT_test, dtype=torch.float32, device=device)
    _video_feature_MSRVTT_test = torch.tensor(_video_feature_MSRVTT_test, dtype=torch.float32, device=device)
    # MSRVTT_train集数据
    # _video_id_MSRVTT_train, _video_feature_MSRVTT_train = TensorDataBase.load(TensorDataBase.MSRVTT_train_video)
    # _video_feature_MSRVTT_train = torch.tensor(_video_feature_MSRVTT_train, dtype=torch.float32, device=device)
    # 在检索结果里附带上video_id_to_sentence.csv文件里储存的句子用于显示
    _video_id_to_sentence = pandas.read_csv(os.path.join(DIR_DataBase, 'video_id_to_sentence.csv')).set_index('video_id') \
        if os.path.exists(os.path.join(DIR_DataBase, 'video_id_to_sentence.csv')) else None

    @classmethod
    def _tokenize(cls, sentences: Union[str, list[str]]):
        """将sentence构造为网络输入text_input"""
        return tokenize(sentences, context_length=args.max_words, truncate=True)

    @classmethod
    def _get_sequence_output(cls, text_input: torch.Tensor):
        """将text_input输入模型获得text_feature、L2正则化后返回"""
        with torch.no_grad():
            sequence_output = cls._model.get_sequence_output(text_input)
            sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)  # L2正则
            return sequence_output

    @classmethod
    def _get_top_n_info(cls, sim_matrix: numpy.ndarray, video_id: numpy.ndarray, top_n: int):
        """返回一个文字-视频相似矩阵中，相似度最大top_n个视频的：[(video_id、sentence、相似度)]"""
        _index = numpy.argsort(-sim_matrix, axis=-1)[:top_n]  # 获得从大到小概率对应的位置（argsort升序，使用负数在结果上等价于降序）
        _video_id = numpy.take_along_axis(video_id, _index, axis=-1)  # top_n的视频名（video_id与sim_matrix一一对应）
        _sentence = list(cls._video_id_to_sentence['sentence'].get(i, '') for i in _video_id
                         ) if cls._video_id_to_sentence is not None else [''] * len(_video_id)
        _sim = numpy.take_along_axis(sim_matrix, _index, axis=-1)  # top_n的sim值
        return zip(_video_id, _sentence, _sim)

    @classmethod
    def search(cls, sentences: Union[str, list[str]], top_n: int, dataset_name: Union[str, list[str]] = 'MSRVTT_test'):
        """
        在指定dataset_name中，用sentences进行文本视频检索，取前top_n个返回\n
        返回 [各sentence相似度最大top_n个视频的：[(video_id（int）、sentence（无则None）、相似度)]]\n
        :param sentences: 要查询的文本，str或list[str]
        :param top_n: 返回的检索结果数量
        :param dataset_name: 在哪些数据集中查找，默认'MSRVTT_test'
        """
        if isinstance(dataset_name, str): dataset_name = list(dataset_name)
        with torch.no_grad():
            # 输入模型获取text_feature
            text_input = cls._tokenize(sentences).to(device)  # [B, args.max_words]
            text_feature = cls._get_sequence_output(text_input)  # 文字特征，[B, 1, 512]
            # 计算相似矩阵，各数据集hstack
            sim_matrix = numpy.hstack([
                cls._model.get_similarity_logits(text_feature, cls.__dict__[f'_video_feature_{i}']).cpu().numpy()
                for i in dataset_name])
            # 各数据集video_id同样hstack
            video_id = numpy.hstack([cls.__dict__[f'_video_id_{i}']
                                     for i in dataset_name])
            return [cls._get_top_n_info(i, video_id, top_n) for i in sim_matrix]


if __name__ == '__main__':
    from Tools import evaluation
    import time

    while 1:
        text = input('请输入要搜索的句子：')
        if text == '0': break
        if text == '1':
            evaluation(VideoSearchSystem._model, device)
            continue
        time1 = time.time()
        results = VideoSearchSystem.search(
            text,
            # ['a woman is demonstrating a nail painting technique',
            #  'a tv channel named how to cook great foodcom is telling how to prepare a dish'],
            top_n=30, dataset_name=['MSRVTT_test'])
        print(f'耗时{time.time() - time1}')
        for infos in results:
            for info in infos: print(f'视频id：{info[0]}， 视频描述：{info[1]:<80}  相似：{info[2]:.3f}')
            print('\n')
