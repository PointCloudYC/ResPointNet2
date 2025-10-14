import yaml
from easydict import EasyDict as edict

config = edict()
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
config.epochs = 600
config.start_epoch = 1
config.base_learning_rate = 0.01
config.lr_scheduler = 'step'  # step,cosine
config.optimizer = 'sgd'
config.warmup_epoch = 5
config.warmup_multiplier = 100
config.lr_decay_steps = 20
config.lr_decay_rate = 0.7
config.weight_decay = 0
config.momentum = 0.9
config.grid_clip_norm = -1
# ---------------------------------------------------------------------------- #
# CB loss
# ---------------------------------------------------------------------------- #
config.use_cb_loss = False
config.cb_beta = 0.9999
config.cb_gamma = 3.0 # 3 for strong version
config.cb_alpha = 2.0 # 2 for strong version
config.cb_loss_type = 'none' # none, standard, ld, gc
# local density aware cb loss
config.ld_density_radius = 0.5
config.ld_density_weight = 1.0
# geometric context aware cb loss
config.gc_context_k = 16
config.gc_context_weight = 1.0
config.gc_entropy_temp = 1.0

# ---------------------------------------------------------------------------- #
# Two-stage training
# ---------------------------------------------------------------------------- #
config.use_two_stage = False
config.stage1_epochs = 200
config.stage2_epochs = 50
config.stage2_lr = 0.001
config.stage2_warmup_epochs = 10
config.sampling_temperature = 2.0
config.use_cb_loss_stage1 = False
config.use_cb_loss_stage2 = True
config.freeze_encoder = True
# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
config.backbone = 'resnet'
config.head = 'resnet_cls'
config.radius = 0.05
config.sampleDl = 0.02
config.density_parameter = 5.0
config.nsamples = []
config.npoints = []
config.width = 144
config.depth = 2
config.bottleneck_ratio = 2
config.bn_momentum = 0.1

# weakly supervision
config.weak_ratio = 0.1

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
config.datasets = 'psnet5'
config.data_root = ''
config.num_classes = 0
config.num_parts = 0
config.input_features_dim = 3
config.batch_size = 32
config.num_points = 5000
config.num_classes = 40
config.num_workers = 4
# data augmentation
config.x_angle_range = 0.0
config.y_angle_range = 0.0
config.z_angle_range = 0.0
config.scale_low = 2. / 3.
config.scale_high = 3. / 2.
config.noise_std = 0.01
config.noise_clip = 0.05
config.translate_range = 0.2
config.color_drop = 0.2
config.augment_symmetries = [0, 0, 0]
# scene segmentation related
config.in_radius = 2.0
config.num_steps = 500

# ---------------------------------------------------------------------------- #
# io and misc
# ---------------------------------------------------------------------------- #
config.load_path = ''
config.print_freq = 10
config.save_freq = 10
config.val_freq = 10
config.log_dir = 'log'
config.local_rank = 0
config.amp_opt_level = ''
config.rng_seed = 0

# ---------------------------------------------------------------------------- #
# Local Aggregation options
# ---------------------------------------------------------------------------- #
config.local_aggregation_type = 'respointnet2'  # respointnet2, pospool, pseudo_grid
# respointnet2
config.respointnet2 = edict()
config.respointnet2.feature_type = 'dp_fj'  # dp_fj, fi_df, dp_fi_df
config.respointnet2.num_mlps = 1
config.respointnet2.reduction = 'max'
# pospool
config.pospool = edict()
config.pospool.position_embedding = 'xyz'  # xyz, sin_cos
config.pospool.reduction = 'avg'  # max, avg, sum
config.pospool.output_conv = False
# pseudo_grid (KPConv)
config.pseudo_grid = edict()
config.pseudo_grid.fixed_kernel_points = 'center'  # center, verticals
config.pseudo_grid.KP_influence = 'linear'  # constant, linear, gaussian
config.pseudo_grid.KP_extent = 1.0
config.pseudo_grid.num_kernel_points = 15
config.pseudo_grid.convolution_mode = 'sum'
config.pseudo_grid.output_conv = False

def update_config(config_file):
    with open(config_file) as f:
        # use safe_load() instead of load() for the new yaml version
        # exp_config = edict(yaml.load(f))
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError(f"{k} key must exist in config.py")
