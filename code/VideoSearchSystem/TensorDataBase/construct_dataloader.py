from Tools import name2index
from preprocess.dataloader_keyFrame import KeyFrameDataset


class MSRVTT_Test_Dataset(KeyFrameDataset):
    """读取Test集所有句子"""

    def __getitem__(self, idx):
        video_id = self.train_data.iloc[idx]['video_id']  # video_id
        if isinstance(video_id, str): video_id = name2index(video_id)
        sentence = self.train_data.iloc[idx]['sentence']  # 句子
        key_frame_count = self.key_frames_data.loc[video_id]['key_frame_count']  # 关键帧数量
        return video_id, self._tokenize(sentence), *self._get_video(video_id, key_frame_count)


class MSRVTT_Train_Dataset_text(KeyFrameDataset):
    """读取Train集所有句子"""

    def __getitem__(self, idx):
        video_id = self.train_data.iloc[idx]['video_id']  # video_id
        if isinstance(video_id, str): video_id = name2index(video_id)
        sentence = self.train_data.iloc[idx]['sentence']  # 句子
        return video_id, self._tokenize(sentence)


class MSRVTT_Train_Dataset_Video(KeyFrameDataset):
    """读取Train集所有视频"""

    def __getitem__(self, idx):
        video_id = self.train_data.iloc[idx]['video_id']  # video_id
        if isinstance(video_id, str): video_id = name2index(video_id)
        key_frame_count = self.key_frames_data.loc[video_id]['key_frame_count']  # 关键帧数量
        return video_id, *self._get_video(video_id, key_frame_count)
