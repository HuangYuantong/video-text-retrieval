import os

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
'''项目根目录'''
DIR_RES = os.path.join(os.path.dirname(DIR_BASE), 'res')
'''res文件夹根目录'''
DIR_MODEL = os.path.join(DIR_RES, 'AIM.pt')
'''模型checkpoint的路径：res文件夹根目录下的CLIP4Clip.pt或AIM.pt'''
DIR_DataBase = os.path.join(DIR_RES, 'TensorDataBase')
'''特征向量数据库：特征向量保存'''

# 视频
dir_MSRVTT_video = os.path.join(DIR_RES, 'MSRVTT', 'videos', 'all')  # MSRVTT视频根目录，下有视频video0.mp4-video9999.mp4
dir_MSRVTT_data = os.path.join(DIR_RES, 'msrvtt_data', 'MSRVTT_data.json')  # 数据集中视频的文字描述
dir_output = os.path.join(DIR_RES, 'outputs')  # 数据输出路径
# 关键帧
# 所有数据集、所有视频关键帧的存储目录（图片命名格式：‘video_id’_‘idx’.jpg）
dir_key_frames_diff = os.path.join(DIR_RES, 'key_frames_diff')  # 最大帧间差采样，其内有 key_frames_data.csv
dir_key_frames_uniform = os.path.join(DIR_RES, 'key_frames_uniform')  # 平均采样，其内有 key_frames_data.csv

# 训练集划分
dir_data_train = os.path.join(DIR_RES, 'train.csv')
dir_MSRVTT_data_train_7k = os.path.join(DIR_RES, 'msrvtt_data', 'MSRVTT_train.7k.csv')  # MSRVTT训练集：7010 [video_id]
dir_MSRVTT_data_train_9k = os.path.join(DIR_RES, 'msrvtt_data', 'MSRVTT_train.9k.csv')  # MSRVTT训练集：9000 [video_id]
# 测试集划分
# MSRVTT测试集：1000 [key, vid_key(msrxxxx), video_id(videoxxxx), sentence]
dir_MSRVTT_data_test = os.path.join(DIR_RES, 'msrvtt_data', 'MSRVTT_JSFUSION_test.csv')


class get_args:
    """所有超参数"""

    # 是否train、eval
    ##################
    do_pretrain = False  # Whether to run training.
    do_train = True  # Whether to run training.
    do_eval = True  # Whether to run eval on the eval set.

    # 路径
    ##################
    train_csv = dir_data_train  # 训练集CSV
    val_csv = dir_MSRVTT_data_test  # 验证集CSV
    dir_key_frames = dir_key_frames_uniform  # 使用的key_frame路径

    # 训练、数据集相关
    ##################
    num_thread_reader = 1  # Dataloader线程数，0在主进程中进行
    lr = 1e-4  # initial learning rate
    epochs = 5  # upper epoch limit
    batch_size = 32  # batch size train
    batch_size_val = 32  # batch size eval, 16、32
    lr_decay = 0.9  # Learning rate exp epoch decay
    n_display = 100  # Information display frequency

    seed = 42  # random seed
    max_words = 77  # 一个句长最多分子词个数
    max_frames = 12  # 一个视频选取最多帧数
    margin = 0.1  # margin for loss
    hard_negative_rate = 0.5  # rate of intra negative sample
    negative_weighting = 1  # Weight the loss for intra negative
    n_pair = 1  # Num of pair to output from data loader

    output_dir = dir_output  # The output directory where the model predictions and checkpoints will be written.
    init_model = None  # type=str, Initial model.
    resume_model = None  # type=str, Resume train model.
    do_lower_case = True  # Set this flag if you are using an uncased model.
    warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass.
    n_gpu = 1  # Changed in the execute process.

    cache_dir = DIR_RES  # Where do you want to store the pre-trained CLIP4Clip downloaded from s3

    task_type = 'retrieval'  # type=str, Point the task `retrieval` to fine-tune.

    world_size = 0  # type = int, distributed training
    local_rank = 0  # type = int, distributed training
    rank = 0  # type = int, distributed training
    coef_lr = 1e-3  # coefficient for bert branch.
    use_mil = False  # Whether use MIL as Miech et. al. (2020).
    sampled_use_mil = False  # Whether MIL, has a high priority than use_mil.

    # 模型相关
    ##################
    text_num_hidden_layers = 12  # Layer NO. of text.
    visual_num_hidden_layers = 12  # Layer NO. of visual.
    freeze_layer_num = 0  # Layer NO. of CLIP need to freeze.
    pretrained_clip_name = 'ViT-B/32'  # type=str, Choose a CLIP version

    # 数据构建相关
    ##################
    expand_msrvtt_sentences = False  # True时以text数量为准构造dataset：同个视频可能出现多次以对齐text数
    # False时以视频数量为准构造dataset：同个视频有多个text待选，dataloader时每个视频从其text中随机选一个参与本epoch
    features_path = dir_MSRVTT_video
    feature_framerate = 1
    train_frame_order = 0  # choices = [0, 1, 2], _Frame_Info order, 0: ordinary order; 1: reverse order; 2: random order.
    eval_frame_order = 0  # choices = [0, 1, 2], _Frame_Info order, 0: ordinary order; 1: reverse order; 2: random order.
    slice_framepos = 2  # choices = [0, 1, 2], 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.

    def __init__(self):
        super(get_args, self).__init__()
        # Check parameters
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f'Invalid gradient_accumulation_steps parameter: {self.gradient_accumulation_steps}, should be >= 1')
        if not self.do_train and not self.do_eval:
            raise ValueError('At least one of `do_train` or `do_eval` must be True.')

        self.batch_size = int(self.batch_size / self.gradient_accumulation_steps)

        # 视频文本检索系统运行，最少所需文件夹
        for i in [DIR_RES, DIR_DataBase, dir_MSRVTT_video, os.path.join(DIR_RES, 'thumbnails')]:
            if not os.path.exists(i): exit(f'文件夹缺失：{i}')
