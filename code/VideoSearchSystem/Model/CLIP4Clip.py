import torch
from torch import nn
from torch.nn import functional as F
from .clip.model import CLIP, convert_weights

from .clip.clip import load
from get_args import DIR_RES


class CrossEn(nn.Module):
    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class CLIP4Clip(nn.Module):
    def __init__(self,
                 embed_dim=512,
                 image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                 context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12,
                 clip_name='ViT-B/32'):
        """
        使用meanP的CLIP4Clip模型  **模型输出的text_output、video_output始终是torch.float32**
        """
        super(CLIP4Clip, self).__init__()
        # 基本CLIP4Clip
        if clip_name == 'ViT-B/32':
            self.clip = CLIP(
                embed_dim,
                image_resolution, vision_layers, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
            convert_weights(self.clip)  # 半精度：conv、Linear、自注意力模块、Transformer最后的矩阵、ViT最后的矩阵
        else:
            self.clip = load(name=clip_name, download_root=DIR_RES)[0]
        self.loss_fct = CrossEn()

    def forward(self, text_input, video, video_mask):
        # video_frame = bs * ts
        sequence_output = self.get_sequence_output(text_input)
        visual_output = self.get_video_output(video)

        # 正则化、视频帧信息融合
        text_features = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        video_features = self.get_meanP_video_features(visual_output, video_mask)
        # 计算余弦相似度、损失
        sim_matrix = self.get_similarity_logits(text_features, video_features)
        sim_loss1 = self.loss_fct(sim_matrix)
        sim_loss2 = self.loss_fct(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        return sim_loss

    def get_sequence_output(self, text):
        """
        计算文本特征
        clip.encode_text: [B, 1, context_length]（1可能没有）->[B, 512]
        """
        text = text.view(-1, text.shape[-1])

        sequence_hidden = self.clip.encode_text(text).float()
        return sequence_hidden

    def get_video_output(self, video):
        """
        计算视频（帧）特征
        clip.encode_image: [B, 1, max_frame, 1, 3,244,244]（1可能没有）->[B, max_frame, 512]
        """
        shape = video.shape
        video = video.view(-1, *shape[-3:])  # [B*max_frame, 3,244,244]

        visual_hidden = self.clip.encode_image(video).float()
        visual_hidden = visual_hidden.view(shape[0], -1, visual_hidden.shape[-1])
        return visual_hidden

    @classmethod
    def get_meanP_video_features(cls, visual_output, video_mask):
        """CLIP4Clip中的meanP方法: [B, 12, 512]->[B, 512]"""
        video_mask_un = video_mask.view(-1, video_mask.shape[-1]).to(dtype=torch.float32).unsqueeze(-1)
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float32)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)  # L2正则，[B, 12, 512]
        visual_output = visual_output * video_mask_un
        visual_output = torch.sum(visual_output, dim=1) / video_mask_un_sum  # [B, 512]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)  # L2正则，[B, 512]
        return visual_output  # [B, 512]

    def get_similarity_logits(self, text_feature, video_feature):
        """
        返回 text_features @ video_features\n
        :param text_feature: 已L2正则化的文本特征
        :param video_feature: 已融合、已L2正则化的视频特征
        """
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = logit_scale * (text_feature @ video_feature.t())
        return logits_per_text
