import numpy
import torch
from torch.utils.data import DataLoader

from .TensorDataBase import TensorDataBase
from .construct_dataloader import MSRVTT_Test_Dataset, MSRVTT_Train_Dataset_text, MSRVTT_Train_Dataset_Video


def construct_database_msrvtt_test(args, tokenizer, model, device: str):
    """构造数据集的features，并保存到database：MSRVTT_test、MSRVTT_test_raw"""
    from tqdm import tqdm
    dataset = MSRVTT_Test_Dataset(args.dir_key_frames, args.val_csv, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size_val)
    print(f'dataset长度{len(dataset)}')
    with torch.no_grad():
        # 暂存所有features数据
        length, batch_size, i = len(dataset), args.batch_size_val, 0
        max_frames, d_model = args.max_frames, model.clip.visual.output_dim
        video_id_list = numpy.zeros([length], int)  # [N,]
        text_feature_list = numpy.zeros([length, d_model], numpy.float32)  # [N, 512]
        video_feature_list = numpy.zeros([length, d_model], numpy.float32)  # [N, 512]
        video_feature_raw_list = numpy.zeros([length, max_frames, d_model], numpy.float32)  # [N, 12, 512]
        model.eval()
        for video_ids, text_input, video_input, video_mask in tqdm(dataloader, mininterval=5):
            # video_ids: [video_id]，该text对应的video_id
            # text_input：[B, 1, [dict(分子词)]]，长度为 max_words，用0填充补齐
            # video_input: [B, 1, max_frames, 1, 3,244,244]，该text对应的视频，max_frames维度用0补齐
            # video_mask: [B, 1, [1,1,...,0,...]]，视频max_frames维度的mask，实际帧处为1
            text_input, video_input, video_mask = text_input.to(device), video_input.to(device), video_mask.to(device)  # 放到运行设备上
            text_features = model.get_sequence_output(text_input)  # 文字特征，[B, 512]
            video_features_raw = model.get_video_output(video_input)  # CLIP输出的视频特征，[B, 12, 512]
            video_features = model.get_meanP_video_features(video_features_raw, video_mask)  # 经过meanP后的视频特征，[B, 512]
            # L2正则、video_features_raw * mask
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            video_features_raw = video_features_raw / video_features_raw.norm(dim=-1, keepdim=True)
            video_features_raw = video_features_raw * \
                                 video_mask.view(-1, video_mask.shape[-1]).to(dtype=torch.float32).unsqueeze(-1)
            video_features_raw = video_features_raw / video_features_raw.norm(dim=-1, keepdim=True)

            # 暂存所有features数据
            video_id_list[i:i + batch_size] = video_ids.cpu().numpy()  # DataLoader导致int被变成tensor([N,])
            text_feature_list[i:i + batch_size] = text_features.cpu().numpy()  # [N, 512]
            video_feature_list[i:i + batch_size] = video_features.cpu().numpy()  # [N, 512]
            video_feature_raw_list[i:i + batch_size] = video_features_raw.cpu().numpy()  # [N, 12, 512]
            i += batch_size
        # 保存
        TensorDataBase.store(TensorDataBase.MSRVTT_test, video_id_list, text_feature_list, video_feature_list)
        TensorDataBase.store(TensorDataBase.MSRVTT_test_raw, video_id_list, text_feature_list, video_feature_raw_list)


def construct_database_msrvtt_text(args, tokenizer, model, device: str):
    """构造数据集的features，并保存到database：MSRVTT_text\n
    在 MSRVTT_TrainDataLoader 中取消视频构建（video和video_mask）以加快速度
    """
    from tqdm import tqdm
    dataset = MSRVTT_Train_Dataset_text(args.dir_key_frames, args.val_csv, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    print(f'dataset长度{len(dataset)}')
    with torch.no_grad():
        # 暂存所有features数据
        length, batch_size, i = len(dataset), args.batch_size, 0
        d_model = model.clip.visual.output_dim
        video_id_list = numpy.zeros([length], int)  # [N,]
        text_feature_list = numpy.zeros([length, d_model], numpy.float32)  # [N, 512]
        model.eval()
        for video_ids, text_input in tqdm(dataloader, mininterval=5):
            # video_ids: [video_id]，该text对应的video_id
            # text_input：[B, 1, [dict(分子词)]]，长度为 max_words，用0填充补齐
            text_input = text_input.to(device)
            text_features = model.get_sequence_output(text_input)  # 文字特征，[B, 512]
            # L2正则、video_features_raw * mask
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 暂存所有features数据
            video_id_list[i:i + batch_size] = video_ids.cpu().numpy()  # DataLoader导致int被变成tensor([N,])
            text_feature_list[i:i + batch_size] = text_features.cpu().data  # [N, 512]
            i += batch_size
        # 保存
        TensorDataBase.store(TensorDataBase.MSRVTT_text, video_id_list, text_feature_list)


def construct_database_msrvtt_train(args, tokenizer, model, device: str):
    """构造数据集的features，并保存到database：MSRVTT_train、MSRVTT_train_raw"""
    from tqdm import tqdm
    dataset = MSRVTT_Train_Dataset_Video(args.dir_key_frames, args.val_csv, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    print(f'dataset长度{len(dataset)}')
    with torch.no_grad():
        # 暂存所有features数据
        length, batch_size, i = len(dataset), args.batch_size, 0
        max_frames, d_model = args.max_frames, model.clip.visual.output_dim
        video_id_list = numpy.zeros([length], int)  # [N,]
        video_feature_list = numpy.zeros([length, d_model], numpy.float32)  # [N, 512]
        video_feature_raw_list = numpy.zeros([length, max_frames, d_model], numpy.float32)  # [N, 12, 512]
        model.eval()
        for video_ids, video_input, video_mask in tqdm(dataloader, mininterval=5):
            # video_ids: [video_id]，该text对应的video_id
            # video_input: [B, 1, max_frames, 1, 3,244,244]，该text对应的视频，max_frames维度用0补齐
            # video_mask: [B, 1, [1,1,...,0,...]]，视频max_frames维度的mask，实际帧处为1
            video, video_mask = video.to(device), video_mask.to(device)  # 放到运行设备上
            video_features_raw = model.get_video_output(video)  # CLIP输出的视频特征，[B, 12, 512]
            video_features = model.get_meanP_video_features(video_features_raw, video_mask)  # 经过meanP后的视频特征，[B, 512]
            # L2正则、video_features_raw * mask
            video_features_raw = video_features_raw / video_features_raw.norm(dim=-1, keepdim=True)
            video_features_raw = video_features_raw * \
                                 video_mask.view(-1, video_mask.shape[-1]).to(dtype=torch.float32).unsqueeze(-1)
            video_features_raw = video_features_raw / video_features_raw.norm(dim=-1, keepdim=True)

            # 暂存所有features数据
            video_id_list[i:i + batch_size] = video_ids.cpu().numpy()  # DataLoader导致int被变成tensor([N,])
            video_feature_list[i:i + batch_size] = video_features.cpu().data  # [N, 512]
            video_feature_raw_list[i:i + batch_size] = video_features_raw.cpu().data  # [N, 12, 512]
            i += batch_size
        # 保存
        TensorDataBase.store(TensorDataBase.MSRVTT_train_video, video_id_list, video_feature_list)
        TensorDataBase.store(TensorDataBase.MSRVTT_train_video_raw, video_id_list, video_feature_raw_list)
