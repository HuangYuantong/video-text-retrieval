import os

import pandas
import torch
from PIL import Image
from pkg_resources import parse_version
from torch.utils.data import Dataset
from torchvision import transforms

from Tools import name2index

text_dtype = torch.long if parse_version(torch.__version__) < parse_version('1.8.0') else torch.int


class KeyFrameDataset(Dataset):
    def __init__(self, dir_key_frames, csv_path, tokenizer, max_words=77, max_frames=12):
        """使用csv文件加载句子、抽取好的关键帧加载图片序列，train和text的csv文件结构相同\n
        返回：text_input, video_input, video_mask\n
        :param dir_key_frames: 抽取好的关键帧目录，其内有 key_frames_data.csv [video_id, key_frame_count]
        :param csv_path: csv文件路径：[video_id, sentence]

         :returns
        ----------

        text_input
            [dict(分子词)]，长度为 max_words，用0填充补齐；
        video_input
            [max_frames,3,244,244]，该text对应的视频，max_frames维度用0补齐；
        video_mask
            [1,1,...,0,...]，视频max_frames维度的mask，实际帧处为1；
        """
        self.train_data = pandas.read_csv(csv_path)  # [video_id, sentence]
        self.key_frames_data = pandas.read_csv(os.path.join(dir_key_frames, 'key_frames_data.csv')
                                               ).set_index('video_id')  # [video_id, key_frame_count]
        self.dir_key_frames = dir_key_frames  # 关键帧目录
        # text
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.sos_token = self.tokenizer.encoder["<|startoftext|>"]  # <sos>，49406
        self.eos_token = self.tokenizer.encoder["<|endoftext|>"]  # <eos>，49407
        # video
        self.max_frames = max_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Image图片默认转为torch.float32
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        video_id = self.train_data.iloc[idx]['video_id']  # video_id
        if isinstance(video_id,str):video_id=name2index(video_id)
        sentence = self.train_data.iloc[idx]['sentence']  # 句子
        key_frame_count = self.key_frames_data.loc[video_id]['key_frame_count']  # 关键帧数量
        return self._tokenize(sentence), *self._get_video(video_id, key_frame_count)

    def _tokenize(self, sentence):
        """将一句话为token序列，用0补齐至max_words"""
        result = [self.sos_token] + self.tokenizer.encode(sentence)[:self.max_words - 2] + [self.eos_token]  # 编码
        result.extend([0] * (self.max_words - len(result)))  # 用0补齐至max_words
        return torch.tensor(result, dtype=text_dtype)

    def _get_video(self, video_id, key_frame_count):
        """从 dir_key_frames 中读取属于 video_id 的 key_frame_count 张图片拼接"""
        images = list(os.path.join(self.dir_key_frames, f'{video_id}_{i}.jpg') for i in range(key_frame_count))  # [video_path]
        images = list(self.transform(Image.open(i)).unsqueeze(0) for i in images)  # [image * key_frame_count]
        video = torch.vstack(images +
                             [torch.zeros(images[0].shape)] * (self.max_frames - key_frame_count))  # 0补齐
        mask = torch.tensor([1] * key_frame_count +
                            [0] * (self.max_frames - key_frame_count), dtype=torch.int)  # [key_frame_count个1, 0...]
        return video, mask
