import os
import numpy
import pandas
from VideoSearchSystem.get_args import DIR_DataBase


class TensorDataBase:
    """
    向量数据库，根据不同数据库名使用相应文件分开存储；\n
    数据库名称关键字：test、raw、train、video、text；\n
    Notes
    -----
     1. 数据库中所有数据使用numpy.ndarray(数字型)存储，load也是返回相应array矩阵，store前请将数据都转为array；\n
     2. 数据库操作目前为pandas，所有DadaFrame相关数据、操作视为对外界不可见，抽象为array的存储和访问；\n
     3. text_feature、video_features_raw已经过L2正则；video_features已经过Tool.meanP_video_features\n
     4. 数据库名称关键字对应数据列：\n
            MSRVTT_test：[video_id, text_feature, video_feature]\n
            MSRVTT_test_raw：[video_id, text_feature, video_feature_raw]\n
            MSRVTT_train_video：[video_id, video_feature]\n
            MSRVTT_train_video_raw：[video_id, video_feature_raw]\n
            MSRVTT_text：[video_id, text_feature]\n
    """
    _columns = ['video_id', 'text_feature', 'video_feature', 'video_feature_raw']

    MSRVTT_test = 'MSRVTT_test'
    '''测试集：只保留meanP后图像特征'''
    MSRVTT_test_raw = 'MSRVTT_test_raw'
    '''测试集：未经meanP的图像特征'''

    MSRVTT_train_video = 'MSRVTT_train_video'
    """训练集：只保留meanP后图像特征"""
    MSRVTT_train_video_raw = 'MSRVTT_train_video_raw'
    '''训练集：未经meanP的图像特征'''
    MSRVTT_text = 'MSRVTT_text'
    '''所有text的特征，一个video_id可能有多个'''

    @classmethod
    def _get_columns(cls, database_name: str):
        """根据数据库名返回相应DataFrame表头list"""
        if database_name.rfind('test') != -1:  # 测试集
            if database_name.rfind('raw') != -1:
                return [cls._columns[0], cls._columns[1], cls._columns[3]]  # ['video_id', 'text_feature', 'video_features_raw']
            else:
                return [cls._columns[0], cls._columns[1], cls._columns[2]]  # ['video_id', 'text_feature', 'video_features']
        elif database_name.rfind('train') != -1:  # 训练集
            if database_name.rfind('raw') != -1:
                return [cls._columns[0], cls._columns[3]]  # ['video_id', 'video_features_raw']
            else:
                return [cls._columns[0], cls._columns[2]]  # ['video_id', 'video_features']
        elif database_name.rfind('text') != -1:  # 所有text的特征，一个video_id可能有多个
            return [cls._columns[0], cls._columns[1]]  # ['video_id', 'text_feature']
        else:
            raise f'get_columns传入参数错误: {database_name}'

    @classmethod
    def _get_dir(cls, database_name: str):
        """根据数据库名返回相应文件路径：database_name.pkl"""
        return os.path.join(DIR_DataBase, f'{database_name}.pkl')

    @classmethod
    def load(cls, database_name: str):
        """从保存的pickle文件中加载相应database\n
        :return: list(numpy.ndarray)"""
        if not os.path.exists(cls._get_dir(database_name)):  # 数据库文件为空
            return [None] * len(cls._get_columns(database_name))
        return (numpy.stack(
            pandas.read_pickle(cls._get_dir(database_name)).reset_index(drop=False)[i], axis=0)
            for i in cls._get_columns(database_name))

    @classmethod
    def store(cls, database_name: str, *args) -> None:
        """保存features到相应pickle文件
        :param database_name: TensorDataBase.数据库名
        :param args: 要保存的特征，按照顺序：video_id(必选)、text_feature、video_features、video_features_raw，只能2个或3个
        """
        for i in args: assert isinstance(i, numpy.ndarray)  # 只能传入numpy.array
        if not isinstance(args[0][0], int):  # video_id必须先用name2index转化为视频编号
            print(f'args[0][0]={args[0][0]}，不是int，异常，尝试继续运行')
        if len(args) == 2:  # database中表头只设置了2、3列两种
            a1, a2 = args
            data = zip(a1, [i for i in a2])
        elif len(args) == 3:
            a1, a2, a3 = args
            data = zip(a1, [i for i in a2], [i for i in a3])
        else:
            raise f'store_dataframe传输numpy数量错误{len(args)}!=2 !=3'
        pandas.DataFrame(columns=cls._get_columns(database_name), data=data
                         ).set_index('video_id').sort_index().to_pickle(cls._get_dir(database_name))  # 按video_id升序
