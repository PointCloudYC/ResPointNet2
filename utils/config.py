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
config.local_aggregation_type = 'respointnet2'  # pospool, continuous_conv
# respointnet2
config.respointnet2 = edict()
config.respointnet2.feature_type = 'dp_fj'  # dp_fj, fi_df, dp_fi_df
config.respointnet2.num_mlps = 1
config.respointnet2.reduction = 'max'
# other types, similar to the structure of the respointnet2

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
