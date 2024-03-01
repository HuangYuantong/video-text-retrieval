import torch
from torch import nn
from functools import partial
from timm.models.layers import trunc_normal_
from einops import rearrange
from .clip.model import ResidualAttentionBlock, VisionTransformer, Transformer


# [ICLR'23] AIM: Adapting Image Models for Efficient Video Action Recognition
# https://github.com/taoyang1122/adapt-image-models


class Adapter(nn.Module):
    def __init__(self, width, mlp_ratio=0.25, skip_connect=True):
        """
        linear下采样 -> 激活层 -> linear上采样\n
        :param mlp_ratio: 下采样率
        :param skip_connect: 是否开启内部的skip连接
        """
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(width * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(width, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, width)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

    def init_weights(self):
        """
        整合自AIM代码中的参数初始化方法
        并将线性层转为半精度（GLUE层无可参数）
        """
        trunc_normal_(self.D_fc1.weight, std=.02)
        nn.init.constant_(self.D_fc1.bias, 0)
        nn.init.constant_(self.D_fc2.weight, 0)
        nn.init.constant_(self.D_fc2.bias, 0)
        for _layer in [self.D_fc1, self.D_fc2]:
            _layer.weight.data = _layer.weight.data.half()
            _layer.bias.data = _layer.bias.data.half()


def _Add_Adapter_to_ResidualAttentionBlock(block: ResidualAttentionBlock, num_frames, width):
    """向ResidualAttentionBlock中添加 T_Adapter_in、T_Adapter、S_Adapter、MLP_Adapter"""
    # # temporal adaptation：T_Adapter_in、T_Adapter（时间，内部不skip连接）
    block.add_module('T_Adapter_in', Adapter(width, skip_connect=True))  # add_module和block.xxx等价
    block.add_module('T_Adapter', Adapter(width, skip_connect=False))
    # spatial adaptation：S_Adapter（空间，内部skip连接）
    block.add_module('S_Adapter', Adapter(width, skip_connect=True))
    # joint adaptation：
    block.add_module('MLP_Adapter', Adapter(width, skip_connect=False))
    # AIM代码中的参数初始化方法
    block.T_Adapter_in.init_weights()
    block.T_Adapter.init_weights()
    block.S_Adapter.init_weights()
    block.MLP_Adapter.init_weights()

    # 重写ResidualAttentionBlock的forward函数
    def _forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=num_frames)
        xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + xt
        # spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.MLP_Adapter(xn)
        return x

    block.forward = partial(_forward, block)


def Add_Adapter_to_ViT(ViT: VisionTransformer, num_frames):
    # ViT内部的d_model（=768）
    width = ViT.transformer.width
    # 在ResidualAttentionBlock中添加Adapter
    for block in ViT.transformer.resblocks:
        _Add_Adapter_to_ResidualAttentionBlock(block, num_frames, width)

    # 在ViT中添加时间编码：[1, num_frames, d_model]
    ViT.register_parameter('temporal_embedding', nn.Parameter(torch.zeros(1, num_frames, width)))

    # 重写ViT的forward函数
    def _forward(self, x: torch.Tensor):
        # x: [B*max_frame, 3,244,244]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(  # 矩阵加：class_embedding会自动在batch上拓展[*, 1, width]
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # <=============== NEW：加上时间编码
        n = x.shape[1]  # [B*max_frame, grid**2+1, width]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=num_frames)
        x = x + self.temporal_embedding.to(x.dtype)
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        # <=============== END NEW
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])  # [B, 197, 768] -> [B, 768]
        if self.proj is not None:  # [B, 768] -> [B, 512]
            x = x @ self.proj
        return x

    ViT.forward = partial(_forward, ViT)
    # 将添加的模块均移到同一设备上
    ViT.to(ViT.conv1.weight.device)


def _Add_Adapter_to_ResidualAttentionBlock_1(block: ResidualAttentionBlock, width):
    """向ResidualAttentionBlock中添加 S_Adapter"""
    # spatial adaptation：S_Adapter（空间，内部skip连接）
    block.add_module('S_Adapter', Adapter(width, skip_connect=True))
    # joint adaptation：
    block.add_module('MLP_Adapter', Adapter(width, skip_connect=False))
    # AIM代码中的参数初始化方法
    block.S_Adapter.init_weights()
    block.MLP_Adapter.init_weights()

    # 重写ResidualAttentionBlock的forward函数
    def _forward(self, x: torch.Tensor):
        # spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.MLP_Adapter(xn)
        return x

    block.forward = partial(_forward, block)


def Add_Adapter_to_Transformer(transformer: Transformer):
    # Transformer内部的d_model（=512）
    width = transformer.width
    # 在ResidualAttentionBlock中添加Adapter
    for block in transformer.resblocks:
        _Add_Adapter_to_ResidualAttentionBlock_1(block, width)
    # 将添加的模块均移到同一设备上
    transformer.to(transformer.resblocks[0].mlp.c_fc.weight.device)
