"""
Distributed evaluating script for scene segmentation with psnet dataset
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
from models import build_scene_segmentation
from datasets import PSNet5Seg, PSNet12Seg
from utils.util import AverageMeter, seg_metrics, sub_seg_metrics, seg_metrics_vis_CM, save_predicted_results_PLY
from utils.logger import setup_logger
from utils.config import config, update_config


def parse_option():
    parser = argparse.ArgumentParser('PSNet5 scene-segmentation evaluating')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--load_path', required=True, type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--dataset_name', type=str, default='psnet5', help='dataset name, support psnet5 or psnet12 etc.')
    parser.add_argument('--log_dir', type=str, default='log_eval', help='log dir [default: log_eval]')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument("--local_rank", type=int, default=0,help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.rng_seed = args.rng_seed

    config.local_rank = args.local_rank
    config.dataset_name = args.dataset_name

    # ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    # config.log_dir = os.path.join(args.log_dir, 'psnet', f'{ddir_name}_{int(time.time())}')
    model_name = args.cfg.split('.')[-2].split('/')[-1] 
    current_time = datetime.now().strftime('%Y%m%d%H%M%S') #20210518221044 means 2021, 5.18, 22:10:44
    # an example is log_dir=log/psnet5/respointnet2_time 
    config.log_dir = os.path.join(args.log_dir, args.dataset_name, f'{model_name}_{int(current_time)}') 

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps:
        config.num_steps = args.num_steps

    print(args)
    print(config)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def get_loader(config):
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    if config.dataset_name =='psnet5':
        val_dataset = PSNet5Seg(input_features_dim=config.input_features_dim,
                            subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                            in_radius=config.in_radius, num_points=config.num_points,
                            num_steps=config.num_steps, num_epochs=20,
                            data_root=config.data_root, transforms=test_transforms,
                            split='val')
    elif config.dataset_name=='psnet12':
        val_dataset = PSNet12Seg(input_features_dim=config.input_features_dim,
                            subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                            in_radius=config.in_radius, num_points=config.num_points,
                            num_steps=config.num_steps, num_epochs=20,
                            data_root=config.data_root, transforms=test_transforms,
                            split='val')
    else:
        raise NotImplementedError("error! The dataset is supported! Change dataset_name to psnet5 or psnet12")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=False)

    return val_loader


def load_checkpoint(config, model):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def main(config):
    val_loader = get_loader(config)
    n_data = len(val_loader.dataset)
    logger.info(f"length of validation dataset: {n_data}")

    model, criterion = build_scene_segmentation(config)
    model.cuda()
    criterion.cuda()

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model)
        logger.info("==> checking loaded ckpt")
        validate('resume', val_loader, model, criterion, config, num_votes=1)# default num_votes 20

    validate('Last', val_loader, model, criterion, config, num_votes=1) # default num_votes 20


def validate(epoch, test_loader, model, criterion, config, num_votes=10):
    """one epoch validating
    Args:
        epoch (int or str): current epoch
        test_loader ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        config ([type]): [description]
        num_votes (int, optional): [description]. Defaults to 10.
    Raises:
        NotImplementedError: [description]
    Returns:
        [int]: mIoU for one epoch over the validation set
    """
    vote_logits_sum = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels] # [item1, item2,...], each item is [C,#sub_pc_pts]
    vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]# [item1, item2,...], each item is [C,#sub_pc_pts]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]# [item1, item2], each item is [1,#sub_pc_pts]
    validation_proj = test_loader.dataset.projections # projections: find the nearest point of orginal pt in the sub-PC, e.g. indices [[22,1000,],...]
    validation_labels = test_loader.dataset.clouds_points_labels
    validation_points = test_loader.dataset.clouds_points
    validation_colors = test_loader.dataset.clouds_points_colors

    val_proportions = np.zeros(config.num_classes, dtype=np.float32) # counts for each category
    for label_value in range(config.num_classes): # elegant code
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
            test_loader.dataset.epoch = v
            points_list=[]
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

                # forward (note: all these are pts/features for sub-pc, as for original pt need the projection var)
                points = points.cuda(non_blocking=True) # BxCx15000 (15000 is the num_points)
                mask = mask.cuda(non_blocking=True) # 15000,
                features = features.cuda(non_blocking=True) # Bx4x15000 (4 is the number of features)
                points_labels = points_labels.cuda(non_blocking=True)  # Bx15000
                cloud_label = cloud_label.cuda(non_blocking=True) # B,
                input_inds = input_inds.cuda(non_blocking=True) # B,15000

                pred = model(points, mask, features) # BxCx15000 (15000 is the num_points)
                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(np.bool) # shape: (15000,) ,original pts 1, masking pts 0
                    logits = pred[ib].cpu().numpy()[:, mask_i] # shape: (C, #original pts), e.g. (5,2943)
                    inds = input_inds[ib].cpu().numpy()[mask_i] # shape: (#orginal pts,), e.g. (2943,)
                    c_i = cloud_label[ib].item() # denote which PC
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits # amazing, collect sum of logits for each pt in sub-pc
                    vote_counts[c_i][:, inds] += 1 # denote how many logits for each pt already
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i] # logits for each pt in the sub-pc

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % config.print_freq == 0:
                    logger.info(
                        f'Test: [{idx}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            subIoUs, submIoU = sub_seg_metrics(config.num_classes, vote_logits,
                                                 test_loader.dataset.sub_clouds_points_labels, val_proportions)
            logger.info(f'E{epoch} V{v} * sub_mIoU {submIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * sub_msIoU {subIoUs}')

            # IoUs, mIoU, overall_acc = seg_metrics(config.num_classes, vote_logits, validation_proj, validation_labels, image_path=IMAGES_PATH, visualize=True, return_OA=True)

            overall_acc = None
            # for last epoch, we will compute more metrics, plot confusion matrix, show results--yc
            if epoch=='Last':
                # define the confusion matrix saving path
                IMAGES_PATH = os.path.join(config.log_dir,"images")
                os.makedirs(IMAGES_PATH, exist_ok=True)

                # compute metircs and plot confusion matrix
                IoUs, mIoU, overall_acc = seg_metrics_vis_CM(
                                                config.num_classes, vote_logits, 
                                                validation_proj, validation_labels,
                                                more_metrics=True,image_path=IMAGES_PATH, 
                                                label_to_names=test_loader.dataset.label_to_names,
                                                visualize_CM=False)

                # save gt and pred results to ply file and show orig/gt/pred results using open3d
                # TODO: uncomment temporarily for evaluting fastly
                # save_predicted_results_PLY(vote_logits, 
                #                            validation_proj,validation_points,
                #                            validation_colors,validation_labels, 
                #                            test_path=config.log_dir, 
                #                            cloud_names=test_loader.dataset.cloud_names,
                #                            open3d_visualize=True)
            else:
                IoUs, mIoU = seg_metrics(config.num_classes, vote_logits, validation_proj, validation_labels)

            logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')
            if overall_acc:
                logger.info(f'E{epoch} V{v}  * OA {overall_acc:.4%}')

    return mIoU


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOAD_DIR"] = os.path.dirname(config.load_path)

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name=f'{config.dataset_name}_eval')
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))

    # main function
    main(config)
