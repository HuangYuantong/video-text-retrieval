import os
import random
import time
import numpy
import torch

from dataloaders.data_dataloaders import DATALOADER_DICT
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from modules.clip.simple_tokenizer import SimpleTokenizer as ClipTokenizer
from modules.optimization import BertAdam
from util import parallel_apply, get_logger

from get_args import get_args
from modules.My_Model import My_Model, Froze

torch.distributed.init_process_group(backend="nccl")

global logger


def set_seed_logger(args: get_args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args: get_args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args: get_args, device):
    """
    froze_model：True：冻结，只训练Adapter，epochs=5时在第2个完成后提前结束，False：不冻结，全调，训练所有epoch；
    load_Clip4Clip：True：加载训练好的CLIP4Clip，False：加载CLIP
    """
    if args.init_model:
        state_dict = torch.load(args.init_model, map_location='cpu')
        dd_adapter_to_vit = 'clip.visual.temporal_embedding' in state_dict
        add_adapter_to_transformer = 'clip.transformer.resblocks.0.MLP_Adapter.D_fc1.weight' in state_dict
        model = My_Model(args.max_frames, dd_adapter_to_vit, add_adapter_to_transformer, args.clip_name)
        model.load_state_dict(state_dict)
        logger.info(f'=> loaded successfully from: "{args.init_model}"')

    else:
        model = My_Model(args.max_frames, args.add_adapter_to_vit, args.add_adapter_to_transformer, args.clip_name)
        logger.info(f'是否在视频编码器里添加Adapter：{args.add_adapter_to_vit}')
        logger.info(f'是否在文字编码器里添加Adapter：{args.add_adapter_to_transformer}')
        # 加载训练好的CLIP4Clip
        if args.load_Clip4Clip:
            model.load_state_dict(torch.load(args.dir_Pretrained_CLIP4Clip, map_location='cpu'), strict=False)
            logger.info(f'=> loaded successfully from: "{args.dir_Pretrained_CLIP4Clip}"')
        else:
            logger.info(f'=> loaded successfully from: "{args.dir_pretrained_CLIP}"')

        # 参数冻结
        if args.froze_model:
            Froze(model, logger)
        else:
            _total = sum(p.numel() for p in model.parameters())
            logger.info(f'无冻结参数，总参数量：{_total / 1000000:.4f}M（{_total}）')

    model.to(device)
    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
    }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_input, video_input, video_mask = batch
        loss = model(text_input, video_input, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=numpy.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=numpy.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list):
    """TODO eval"""
    sim_matrix = []
    for idx1, sequence_output in enumerate(batch_sequence_output_list):
        each_row = []
        for idx2, visual_output in enumerate(batch_visual_output_list):
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits = model.get_similarity_logits(sequence_output, visual_output).cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = numpy.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu):
    """TODO eval"""
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    # below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text_input, video_input, video_mask = batch

            if not multi_sentence_:
                text_features = model.get_sequence_output(text_input)  # 文字特征，[B, 512]
                video_features_raw = model.get_video_output(video_input)  # CLIP输出的视频特征，[B, 12, 512]
                # L2正则、video_features_raw * mask
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                video_features = model.get_meanP_video_features(video_features_raw,
                                                                video_mask)  # 经过meanP后的视频特征，[B, 512]

                batch_sequence_output_list.append(text_features)
                batch_visual_output_list.append(video_features)

            else:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b = video_input.shape[0]
                text_features = model.get_sequence_output(text_input)
                # L2正则、video_features_raw * mask
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                batch_sequence_output_list.append(text_features)

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if s_ <= itm < e_]
                if len(filter_inds) > 0:
                    video_input, video_mask = video_input[filter_inds, ...], video_mask[filter_inds, ...]
                    video_features_raw = model.get_video_output(video_input)
                    video_features = model.get_meanP_video_features(video_features_raw, video_mask)
                    batch_visual_output_list.append(video_features)
                total_video_num += b

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = numpy.concatenate(tuple(sim_matrix), axis=0)

    if not multi_sentence_:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    else:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(numpy.concatenate((sim_matrix[s_:e_],
                                                     numpy.full((max_length - e_ + s_, sim_matrix.shape[1]), -numpy.inf)),
                                                    axis=0))
        sim_matrix = numpy.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
            format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1


def main(args: get_args):
    global logger
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device)
    # # TODO 使用原始CLIP
    # if load_clip:
    #     model.clip = load(args.dir_pretrained_CLIP)[0]
    #     logger.info(f'=> loaded successfully from: "{args.dir_pretrained_CLIP}"')

    ######################################
    # dataloader loading
    ######################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    # TODO 直接用test做验证集
    if test_dataloader is not None:
        val_dataloader, val_length = test_dataloader, test_length

    # report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    #####################################
    # train and eval
    #####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                     args.local_rank,
                                                     coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ################################################################
        # resume optimizer state besides loss to continue train
        ################################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch'] + 1
            resumed_loss = checkpoint['loss']

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                # Run on val dataset, this process is *TIME-consuming*.
                logger.info("Eval on val dataset")
                R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

                # TODO 损失nan，提前终止
                if torch.isnan(torch.tensor(tr_loss)):
                    logger.info('遇到nan，提前终止本次训练')
                    return -1
                # TODO 网络冻结时，总是在第2个epoch结束后达到最优，提前终止
                if args.froze_model and 0 < args.stop_at_epoch <= epoch + 1:
                    logger.info(f'在第{args.stop_at_epoch}个epoch结束后终止')
                    return best_score

        # Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)
        return best_score

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)


if __name__ == "__main__":
    from get_args import dir_key_frames_diff, dir_key_frames_uniform, dir_output, DIR_RES

    _args = get_args()
    _args.lr = 1e-4  # 学习率
    _args.coef_lr = 1e-3  # CLIP内部学习率 = lr*coef_lr
    # _args.batch_size = 64  # batch size train
    # _args.batch_size_val = 64  # batch size eval, 16、32
    # # 添加ViT，CLIP4Clip为基础冻结，epoch=5
    # _args.output_dir = os.path.join(dir_output, '3_1_CLIP4Clip42_ViT')
    # _args.add_adapter_to_transformer = False
    # _args.load_Clip4Clip = True
    # _args.froze_model = True
    # score = main(_args)
    #
    # if score < 43.2:
    #     _args.batch_size = 32  # batch size train
    #     _args.batch_size_val = 32  # batch size eval, 16、32
    #     logger.info('因为batch size = 64时比32差，batch size修改为32')
    #     _args.output_dir = os.path.join(dir_output, '3_1_CLIP4Clip42_ViT_1')
    #     main(_args)
    #
    # # 添加ViT和text，CLIP4Clip为基础冻结，epoch=5
    # _args.output_dir = os.path.join(dir_output, '3_2_CLIP4Clip42_ViT_text')
    # _args.add_adapter_to_transformer = True
    # _args.load_Clip4Clip = True
    # _args.froze_model = True
    # main(_args)
    # # 添加ViT，CLIP4Clip为基础冻结，epoch=5
    # _args.output_dir = os.path.join(dir_output, '3_3_CLIP4Clip42_ViT_lr5e-5')
    # _args.add_adapter_to_transformer = False
    # _args.load_Clip4Clip = True
    # _args.froze_model = True
    # _args.lr = 5e-5  # 学习率
    # main(_args)
    # # 添加ViT，CLIP4Clip为基础冻结，epoch=5
    # _args.output_dir = os.path.join(dir_output, '3_4_CLIP4Clip42_ViT_lr1e-5')
    # _args.add_adapter_to_transformer = False
    # _args.load_Clip4Clip = True
    # _args.froze_model = True
    # _args.lr = 1e-5  # 学习率
    # main(_args)

    # _args.data_path = '/media/dc/新加卷3/dc/dataset/MSVD'
    # _args.features_path = '/media/dc/新加卷3/dc/dataset/MSVD/videos'
    # _args.datatype = 'msvd'
    # _args.feature_framerate = 1
    # _args.num_thread_reader = 2
    # _args.batch_size_val = 16
    # # 测试AIM
    # _args.do_train = False
    # _args.output_dir = os.path.join(dir_output, '1_MSVD')
    # _args.init_model = '/media/dc/新加卷3/dc/hyt/hyt2/res/outputs4：添加ViT，42.2为基础，最佳得分测试/3_3_CLIP4Clip42_ViT_lr5e-5' \
    #                    '/pytorch_model.bin.1'
    # main(_args)
    # # 测试CLIP4Clip
    # _args.init_model = _args.dir_Pretrained_CLIP4Clip
    # main(_args)
    # # 添加ViT，CLIP4Clip为基础冻结，epoch=5，第2个epoch完成后结束
    # _args.do_train = True
    # _args.init_model = False
    # _args.output_dir = os.path.join(dir_output, '2_1_MSVD_CLIP4Clip_ViT')
    # _args.add_adapter_to_transformer = False
    # _args.load_Clip4Clip = True
    # _args.froze_model = True
    # _args.stop_at_epoch = 2
    # main(_args)
    # # 添加ViT，CLIP为基础冻结，epoch=5，第2个epoch完成后结束
    # _args.output_dir = os.path.join(dir_output, '2_2_MSVD_CLIP_ViT')
    # _args.add_adapter_to_transformer = False
    # _args.load_Clip4Clip = False
    # _args.froze_model = True
    # _args.stop_at_epoch = 2
    # main(_args)

    # 在保存的关键帧上上测试最初的原始CLIP4Clip
    # _args.do_train = False
    # _args.output_dir = os.path.join(dir_output, '4_0_原始Clip4Clip')
    # _model_list = [
    #     '/media/dc/新加卷/dc/CLIP4Clip-master/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.0',
    #     '/media/dc/新加卷/dc/CLIP4Clip-master/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.1',
    #     '/media/dc/新加卷/dc/CLIP4Clip-master/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.2',
    #     '/media/dc/新加卷/dc/CLIP4Clip-master/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.3',
    #     '/media/dc/新加卷/dc/CLIP4Clip-master/ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4',
    # ]
    # for i in _model_list:
    #     _args.init_model = i
    #     main(_args)

    # # CLIP4Clip在uniform上的全微调，DataLoader不shuffle，train.csv用shuffle
    # _args.do_train = True
    # _args.load_Clip4Clip = False  # 以CLIP为起点
    # _args.add_adapter_to_vit = False  # 原始CLIP
    # _args.froze_model = False  # 全微调
    # _args.output_dir = os.path.join(dir_output, '4_1_CLIP4Clip_uniform')
    # _args.shuffle = False
    # score1 = main(_args)
    # # CLIP4Clip在uniform上的全微调，DataLoader用shuffle，train.csv不shuffle
    # _args.output_dir = os.path.join(dir_output, '4_2_CLIP4Clip_uniform_shuffle')
    # _args.shuffle = True
    # main(_args)
    # # CLIP4Clip在diff上的全微调
    # _args.dir_key_frames = dir_key_frames_diff
    # _args.output_dir = os.path.join(dir_output, '4_3_CLIP4Clip_diff' if score1 > score2 else '4_3_CLIP4Clip_diff_shuffle')
    # _args.shuffle = False if score1 > score2 else True
    # main(_args)
    # # CLIP为基础，训练Adapter和Transformer
    # _args.load_Clip4Clip = False  # 以CLIP为起点
    # _args.add_adapter_to_vit = True
    # _args.froze_model = True
    # _args.stop_at_epoch = 2
    # _args.output_dir = os.path.join(dir_output, '4_3_CLIP_AIM_Transformer')
    # main(_args)
    # # CLIP为基础，训练训练ViT
    # _args.add_adapter_to_vit = False
    # _args.froze_model = True
    # _args.stop_at_epoch = 2
    # _args.output_dir = os.path.join(dir_output, '4_4_CLIP_ViT')
    # main(_args)
    _args.do_train = True
    _args.stop_at_epoch = 2
    _args.add_adapter_to_transformer = True
    _args.add_adapter_to_vit = True
    # CLIP为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = False
    _args.output_dir = os.path.join(dir_output, '4_5_CLIP_ViT_Transformer')
    main(_args)
    # CLIP4Clip 41.7为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = True
    _args.dir_Pretrained_CLIP4Clip = os.path.join(DIR_RES, 'CLIP4Clip.pt')
    _args.output_dir = os.path.join(dir_output, '4_6_CLIP4Clip41.7_ViT_Transformer')
    main(_args)
    # CLIP4Clip 42.2为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = True
    _args.dir_Pretrained_CLIP4Clip = os.path.join(DIR_RES, 'CLIP4Clip.pt42.2')
    _args.output_dir = os.path.join(dir_output, '4_7_CLIP4Clip42.2_ViT_Transformer')
    main(_args)

    _args.lr = 5e-5
    # CLIP为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = False
    _args.output_dir = os.path.join(dir_output, '4_5_CLIP_ViT_Transformer_5e-5')
    main(_args)
    # CLIP4Clip 41.7为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = True
    _args.dir_Pretrained_CLIP4Clip = os.path.join(DIR_RES, 'CLIP4Clip.pt')
    _args.output_dir = os.path.join(dir_output, '4_6_CLIP4Clip41.7_ViT_Transformer_5e-5')
    main(_args)
    # CLIP4Clip 42.2为基础，在Transformer和ViT添加Adapter，冻结
    _args.load_Clip4Clip = True
    _args.dir_Pretrained_CLIP4Clip = os.path.join(DIR_RES, 'CLIP4Clip.pt42.2')
    _args.output_dir = os.path.join(dir_output, '4_7_CLIP4Clip42.2_ViT_Transformer_5e-5')
    main(_args)
'''
python -m torch.distributed.launch --nproc_per_node=1 main_task_retrieval.py
'''
