import torch
from .CLIP4Clip import CLIP4Clip
from .AIM import Add_Adapter_to_ViT, Add_Adapter_to_Transformer


class My_Model(CLIP4Clip):
    def __init__(self, num_frames: int, add_adapter_to_vit: bool, add_adapter_to_transformer: bool, clip_name: str = 'ViT-B/32'):
        """
        创建半浮点、默认未初始化的模型；\n
        不添加任何Adapter时，即为使用meanP的CLIP4Clip；\n
        **模型输出的text_output、video_output始终是torch.float32；**\n
        :param num_frames: 视频的采样帧数（args.max_frames）
        :param add_adapter_to_vit: 在视频编码器中插入Adapter
        :param add_adapter_to_transformer: True：在文字编码器中插入Adapter
        """
        super(My_Model, self).__init__(clip_name=clip_name)
        # 在ViT中添加Adapter（T、S、MLP）、时间编码
        if add_adapter_to_vit:
            Add_Adapter_to_ViT(self.clip.visual, num_frames)
        # 在Transformer的自注意旁添加一个Adapter
        if add_adapter_to_transformer:
            Add_Adapter_to_Transformer(self.clip.transformer)


def load_model(model_path: str, device: str, max_frames: int = None, logger=None, model: My_Model = None):
    """
    从model_path加载state_dict，并构建相应模型（CLIP4Clip、添加Adapter），返回model.eval()\n
    :param model_path: state_dict路径，可以是CLIP、CLIP4Clip、AIM
    :param device: 运行设备
    :param max_frames: 视频关键帧数T，Adapter的额外参数
    :param logger: 记录日志
    :param model: 模型，state_dict加载到model中，None则自动创建最符合state_dict的模型
    """
    infos = list()
    # 加载state_dict
    try:  # Just In Time Compilation 即时编译（程序优化），JIT模型TorchScript Module是静态图
        state_dict = torch.jit.load(model_path, map_location='cpu').eval().state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    # 若没有以’clip.‘开头的模型则认为是加载原始CLIP参数，所有key加上前缀’clip.‘
    if all(not i.startswith('clip.') for i in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            new_state_dict['clip.' + key] = val.clone()
        del state_dict
        state_dict = new_state_dict

    # 根据state_dict构造模型
    if not model:
        add_adapter_to_vit = any('clip.visual' in i and 'Adapter' in i for i in state_dict)
        add_adapter_to_transformer = any('clip.transformer' in i and 'Adapter' in i for i in state_dict)
        model = My_Model(max_frames, add_adapter_to_vit, add_adapter_to_transformer)
        infos.append(f'=> 视频编码器中插入Adapter：{add_adapter_to_vit}，文字编码器里插入Adapter：{add_adapter_to_transformer}')

    # 加载state_dict
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys: infos.append(f'missing_keys: {msg.missing_keys}')
    if msg.unexpected_keys: infos.append(f'unexpected_keys: {msg.unexpected_keys}')
    infos.append(f'=> loaded successfully from: "{model_path}"')
    [logger.info(info) if logger is not None else print(info) for info in infos]

    if str(device) == 'cpu': model.float()  # CLIP代码中，若运行设备是CPU，则使用浮点torch.float32
    model.eval()
    return model.to(device)


def Froze(model: torch.nn.Module, logger=None):
    """
    将模型部分冻结，可训练参数如下：
    若clip.visual中插入了Adapter：
        Adapter、时间编码（temporal_embedding）、ViT的最后一个LayerNorm（ln_post）
    若clip.transformer中插入了Adapter：
        Adapter、CLIP中用于文本编码器的最后一个LayerNorm（ln_final）
    """
    _keys = model.state_dict().keys()
    _unfroze_param = ['Adapter']
    # 模型在视频编码器（clip.visual）中插入了Adapter
    if any('clip.visual' in i and 'Adapter' in i for i in _keys):
        _unfroze_param.extend(['temporal_embedding', 'ln_post'])
    # 模型在文本编码器（clip.transformer）中插入了Adapter
    if any('clip.transformer' in i and 'Adapter' in i for i in _keys):
        _unfroze_param.extend(['ln_final'])

    # 不在_unfroze_param中的参数全部冻结
    for name, param in model.named_parameters():
        if all(i not in name for i in _unfroze_param):
            param.requires_grad = False

    # 输出模型中可训练参数数量
    _tunable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    info = f'可调参数量：{_tunable / 1000000:.4f}M（{_tunable}），总参数量：{_total / 1000000:.4f}M（{_total}），' \
           f'占比：{100 * (_tunable / _total):.4f}%'
    logger.info(info) if logger is not None else print(info)
    if _tunable == 0:
        info = '****** Frozen() warning：模型中未添加任何Adapter，模型已被全部冻结 ******'
        logger.warning(info) if logger is not None else print(info)
