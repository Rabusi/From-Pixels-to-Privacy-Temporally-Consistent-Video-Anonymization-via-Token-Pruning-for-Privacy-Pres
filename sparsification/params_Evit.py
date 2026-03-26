
# Job parameters.
run_id = 'dual_attn_Evit_vit_b_prune_0.9_lambda_priv_0.5_vphmdb'

# Dataset parameters.
num_action = 51
num_priv = 5
data_percentage = 1.0
fix_skip = 2
num_modes = 5
num_skips = 1


# Transformer parameters.
img_size = 224
patch_size = 16
tubelet_size = 2
all_frames = 16
in_chans = 3
embed_dim = 768
depth = 12
num_heads = 12
mlp_ratio = 4.0
drop_rate = 0.2
attn_drop_rate = 0.1
drop_path_rate = 0.2
distill = False
keep_rate = (1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0)
keep_rate_eval = (1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0)
fuse_token = True


# Training parameters.
batch_size = 8
num_workers = 4
learning_rate = 1e-4
weight_decay = 0.05
num_epochs = 80

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
weak_aug = False
no_ar_distortion = False
aspect_ratio_aug = False

# Training augmentation params.
reso_h = 224
reso_w = 224
min_crop_factor_training = 0.6
temporal_align = False


# Anonymization training parameters.
lambda_act       = 1.0

# Tracking params.
wandb = False
