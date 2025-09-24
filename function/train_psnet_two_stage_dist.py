"""
Two-Stage Distributed training script for ResPointNet-LT scene segmentation
Implements decoupled representation learning and classifier fine-tuning.
"""
import argparse
import os
import sys
import time
import json
import random
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import datasets.data_utils as d_utils
from models import build_scene_segmentation, build_scene_segmentation_with_cb_loss
from datasets import PSNet5Seg, PSNet12Seg
from utils.util import AverageMeter, seg_metrics, sub_seg_metrics, seg_part_metrics
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from utils.config import config, update_config
from LTL.two_stage_trainer import create_two_stage_trainer


def parse_option():
    parser = argparse.ArgumentParser('PSNet Two-Stage Training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--dataset_name', type=str, default='psnet12', help='dataset name, support psnet5 or psnet12 etc.')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')

    # Two-stage specific parameters
    parser.add_argument('--stage1_epochs', type=int, help='number of stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, help='number of stage 2 epochs')
    parser.add_argument('--stage2_lr', type=float, help='stage 2 learning rate')
    parser.add_argument('--use_cb_loss_stage1', action='store_true', help='use CB loss in stage 1')
    parser.add_argument('--use_cb_loss_stage2', action='store_true', help='use CB loss in stage 2')
    parser.add_argument('--freeze_encoder', action='store_true', default=True, help='freeze encoder in stage 2')

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='patience for early stopping')

    # CB loss parameters
    parser.add_argument('--use_cb_loss', action='store_true', help='use class-balanced loss')
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='CB loss beta parameter')
    parser.add_argument('--cb_gamma', type=float, default=2.0, help='CB loss gamma (focal) parameter')
    parser.add_argument('--cb_alpha', type=float, default=1.0, help='CB loss alpha (weighting) parameter')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    config.data_root = args.data_root
    config.dataset_name = args.dataset_name
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed
    config.local_rank = args.local_rank
    config.early_stopping_patience = args.early_stopping_patience
    
    # Two-stage specific configurations
    config.stage1_epochs = args.stage1_epochs or config.epochs
    config.stage2_epochs = args.stage2_epochs or (config.epochs // 4)
    config.stage2_lr = args.stage2_lr or (config.base_learning_rate * 0.1)
    config.use_cb_loss_stage1 = args.use_cb_loss_stage1
    config.use_cb_loss_stage2 = args.use_cb_loss_stage2 or True  # Default to True for stage 2
    config.freeze_encoder = args.freeze_encoder
    
    # CB loss parameters
    config.use_cb_loss = args.use_cb_loss
    config.cb_beta = args.cb_beta
    config.cb_gamma = args.cb_gamma
    config.cb_alpha = args.cb_alpha

    # Handle log directory - if specific path provided, use it directly
    if hasattr(args, 'log_dir') and args.log_dir and args.log_dir != 'log':
        # If it's an absolute path or contains the experiment name, use it directly
        if os.path.isabs(args.log_dir) or 'two_stage' in args.log_dir:
            config.log_dir = args.log_dir
        else:
            # Otherwise treat as base directory and create timestamped subdirectory
            model_name = args.cfg.split('.')[-2].split('/')[-1] 
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            config.log_dir = os.path.join(args.log_dir, args.dataset_name, f'{model_name}_two_stage_{current_time}')
    elif args.load_path:
        config.log_dir = args.log_dir
    else:
        model_name = args.cfg.split('.')[-2].split('/')[-1] 
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        config.log_dir = os.path.join(args.log_dir, args.dataset_name, f'{model_name}_two_stage_{current_time}')

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    print(args)
    print(config)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def get_loader(config):
    # set the data loader
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range),
        d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries),
    ])

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
    ])
    
    if config.dataset_name =='psnet5':
        train_dataset = PSNet5Seg(input_features_dim=config.input_features_dim,
                                subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=config.epochs,
                                data_root=config.data_root, transforms=train_transforms,
                                split='train')
        val_dataset = PSNet5Seg(input_features_dim=config.input_features_dim,
                            subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                            in_radius=config.in_radius, num_points=config.num_points,
                            num_steps=config.num_steps, num_epochs=20,
                            data_root=config.data_root, transforms=test_transforms,
                            split='val')
    elif config.dataset_name=='psnet12':
        train_dataset = PSNet12Seg(input_features_dim=config.input_features_dim,
                                subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=config.epochs,
                                data_root=config.data_root, transforms=train_transforms,
                                split='train')
        val_dataset = PSNet12Seg(input_features_dim=config.input_features_dim,
                            subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                            in_radius=config.in_radius, num_points=config.num_points,
                            num_steps=config.num_steps, num_epochs=20,
                            data_root=config.data_root, transforms=test_transforms,
                            split='val')
    else:
        raise NotImplementedError("error! The dataset is supported! Change dataset_name to psnet5 or psnet12")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=False)

    return train_loader, val_loader


def validate(epoch, test_loader, model, criterion, runing_vote_logits, config, num_votes=10):
    """one epoch validating
    Args:
        epoch (int or str): current epoch
        test_loader ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        runing_vote_logits ([type]): [description]
        config ([type]): [description]
        num_votes (int, optional): [description]. Defaults to 10.
    Raises:
        NotImplementedError: [description]
    Returns:
        [int]: mIoU for one epoch over the validation set
    """

    vote_logits_sum = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]
    vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]
    validation_proj = test_loader.dataset.projections
    validation_labels = test_loader.dataset.clouds_points_labels
    test_smooth = 0.95

    val_proportions = np.zeros(config.num_classes, dtype=np.float32)
    for label_value in range(config.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.clouds_points_labels])

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        RT = d_utils.BatchPointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                                 z_range=config.z_angle_range)
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                                   std=config.noise_std, clip=config.noise_clip,
                                                   augment_symmetries=config.augment_symmetries)
        for v in range(num_votes):
            # dealing w. distributed training to resolve the "NCCL operations have failed or timed out"
            if dist.get_world_size() > 1:
                dist.barrier()
            test_loader.dataset.epoch = (0 + v) if isinstance(epoch, str) else (epoch + v) % 20
            predictions = []
            targets = []
            for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(test_loader):
                # augment for voting
                if v > 0:
                    points = RT(points)
                    points = TS(points)
                    if config.input_features_dim <= 5:
                        pass
                    elif config.input_features_dim == 6:
                        color = features[:, :3, :]
                        features = torch.cat([color, points.transpose(1, 2).contiguous()], 1)
                    elif config.input_features_dim == 7:
                        color_h = features[:, :4, :]
                        features = torch.cat([color_h, points.transpose(1, 2).contiguous()], 1)
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.input_features_dim} in voting not supported")
                # forward
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                cloud_label = cloud_label.cuda(non_blocking=True)
                input_inds = input_inds.cuda(non_blocking=True)

                pred = model(points, mask, features)
                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]
                    runing_vote_logits[c_i][:, inds] = test_smooth * runing_vote_logits[c_i][:, inds] + \
                                                       (1 - test_smooth) * logits
                    predictions += [logits]
                    targets += [test_loader.dataset.sub_clouds_points_labels[c_i][inds]]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % config.print_freq == 0:
                    logger.info(
                        f'Test: [{idx}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            pIoUs, pmIoU = seg_part_metrics(config.num_classes, predictions, targets, val_proportions)

            logger.info(f'E{epoch} V{v} * part_mIoU {pmIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * part_msIoU {pIoUs}')

            runsubIoUs, runsubmIoU = sub_seg_metrics(config.num_classes, runing_vote_logits,
                                                       test_loader.dataset.sub_clouds_points_labels, val_proportions)
            logger.info(f'E{epoch} V{v} * running sub_mIoU {runsubmIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * running sub_msIoU {runsubIoUs}')

            subIoUs, submIoU = sub_seg_metrics(config.num_classes, vote_logits,
                                                 test_loader.dataset.sub_clouds_points_labels, val_proportions)
            logger.info(f'E{epoch} V{v} * sub_mIoU {submIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * sub_msIoU {subIoUs}')

            IoUs, mIoU = seg_metrics(config.num_classes, vote_logits, validation_proj, validation_labels)
            logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')

        # dealing w. distributed training to resolve the "NCCL operations have failed or timed out"
        if dist.get_world_size() > 1:
            dist.barrier()

    return mIoU


def main(config):
    train_loader, val_loader = get_loader(config)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    n_data = len(val_loader.dataset)
    logger.info(f"length of validation dataset: {n_data}")

    # Build model and loss function
    if config.use_cb_loss:
        logger.info("Using Class-Balanced Loss")
        model, criterion = build_scene_segmentation_with_cb_loss(config, train_loader.dataset)
    else:
        logger.info("Using standard MaskedCrossEntropy Loss")
        model, criterion = build_scene_segmentation(config)
    
    model.cuda()
    criterion.cuda()

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.batch_size * dist.get_world_size() / 8 * config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.base_learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.base_learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported")

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    # add find_unused_parameters=True to overcome the error "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"
    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    # Create two-stage trainer
    trainer = create_two_stage_trainer(config, model, criterion, optimizer, scheduler,
                                     train_loader, val_loader, logger)

    runing_vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                          val_loader.dataset.sub_clouds_points_labels]

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        trainer._load_checkpoint(config.load_path)
        logger.info("==> checking loaded ckpt")
        validate('resume', val_loader, model, criterion, runing_vote_logits, config, num_votes=2)

    # Stage 1: Representation Learning
    logger.info("Starting Two-Stage Training")
    if not trainer.stage1_completed:
        trainer.train_stage1(lambda epoch, val_loader, model, criterion, num_votes: 
                            validate(epoch, val_loader, model, criterion, runing_vote_logits, config, num_votes))
    else:
        logger.info("Stage 1 already completed, skipping to Stage 2")

    # Stage 2: Classifier Re-training
    trainer.train_stage2(lambda epoch, val_loader, model, criterion, num_votes:
                        validate(epoch, val_loader, model, criterion, runing_vote_logits, config, num_votes))

    # Final validation
    validate('Final', val_loader, model, criterion, runing_vote_logits, config, num_votes=20)


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # add for debugging
    torch.autograd.set_detect_anomaly(True)

    os.makedirs(config.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir

    # Check if we're resuming (Stage 1 checkpoint exists)
    stage1_checkpoint = os.path.join(config.log_dir, 'stage1', 'stage1_best.pth')
    is_resuming = os.path.exists(stage1_checkpoint)
    
    if is_resuming:
        logger = setup_logger(output=os.path.join(config.log_dir, "log_resume.txt"), 
                            distributed_rank=dist.get_rank(), name=f'{config.dataset_name}_two_stage')
        if dist.get_rank() == 0:
            logger.info("=" * 60)
            logger.info("RESUMING TWO-STAGE TRAINING")
            logger.info(f"Found Stage 1 checkpoint: {stage1_checkpoint}")
            logger.info("=" * 60)
    else:
        logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name=f'{config.dataset_name}_two_stage')
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))
    main(config)
