import torch
from .CLIP4Clip import CLIP4Clip
from .AIM import Add_Adapter_to_ViT, Add_Adapter_to_Transformer


class My_Model(CLIP4Clip):
    def __init__(self, num_frames: int, add_adapter_to_vit: bool, add_adapter_to_transformer: bool = False, clip_name='ViT-B/32'):
        """
        添加AIM后的CLIP4Clip模型  **模型输出的text_output、video_output始终是torch.float32**\n
        :param num_frames: 视频的采样帧数（args.max_frames）
        :param add_adapter_to_vit: 在视频编码器中插入Adapter
        :param add_adapter_to_transformer: True：在文字编码器里添加Adapter
        """
        super(My_Model, self).__init__(clip_name=clip_name)
        # 在ViT中添加Adapter（T、S、MLP）、时间编码
        if add_adapter_to_vit:
            Add_Adapter_to_ViT(self.clip.visual, num_frames)
        # 在Transformer的自注意旁添加一个Adapter
        if add_adapter_to_transformer:
            Add_Adapter_to_Transformer(self.clip.transformer)


def load_model(model_file: str, device: str, max_frames: int = None, logger=None):
    """
    加载state_dict\n
    :param model_file: state_dict路径
    :param device: 运行设备
    :param max_frames: 添加Adapter的额外参数
    """
    state_dict = torch.load(model_file, map_location='cpu')
    add_adapter_to_vit = 'clip.visual.temporal_embedding' in state_dict
    add_adapter_to_transformer = 'clip.transformer.resblocks.0.MLP_Adapter.D_fc1.weight' in state_dict
    model = My_Model(max_frames, add_adapter_to_vit, add_adapter_to_transformer)
    model.load_state_dict(state_dict, strict=False)
    if logger is not None:
        logger.info(f'=> loaded successfully from: "{model_file}"')
    if str(device) == 'cpu': model.float()
    return model.to(device)


def Froze(model, logger=None):
    """
    可训练参数：
    AIM：时间编码（temporal_embedding）、Adapter、ViT原有的最后一个LayerNorm（ln_post）
    """
    for name, param in model.named_parameters():
        if 'temporal_embedding' not in name and 'ln_post' not in name and 'Adapter' not in name \
                and 'ln_final' not in name and 'logit_scale' not in name:
            param.requires_grad = False
        # # 训练Adapter和Transformer
        # if temp_mark.endswith('4_3_CLIP_AIM_Transformer'):
        #     if 'clip.visual' in name:
        #         param.requires_grad = False
        #     if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name:
        #         param.requires_grad = True
        # # 训练ViT
        # if temp_mark.endswith('4_4_CLIP_ViT'):
        #     if 'ln_post' not in name and 'logit_scale' not in name and 'clip.visual' not in name:
        #         param.requires_grad = False

    _tunable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    info = f'可调参数量：{_tunable / 1000000:.4f}M（{_tunable}），总参数量：{_total / 1000000:.4f}M（{_total}），' \
           f'占比：{100 * (_tunable / _total):.4f}%'
    if logger is not None: logger.info(info)
