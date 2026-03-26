import os.path
from segmentation_models_pytorch import UnetPlusPlus
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.video import r3d_18, mvit_v2_s
import logging

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparsification.Evit_new import EViT
# from action_training import params_action as params  # for raw action training
from action_training import params_anonymized_action as params   # for anonymized action training

from aux_code.models.i3d import InceptionI3d
from aux_code.models.large_i3d import I3Res50


# Fs model loading function.
def load_fs_model(saved_model_file=None):
    fs_model = EViT(
        img_size=params.img_size,
        patch_size=params.patch_size,
        in_chans=params.in_chans,
        num_classes=params.num_action,
        num_priv=params.num_priv,
        embed_dim=params.embed_dim,
        depth=params.depth,
        num_heads=params.num_heads,
        mlp_ratio=params.mlp_ratio,
        keep_rate=params.keep_rate,
        drop_rate=params.drop_rate,
        attn_drop_rate=params.attn_drop_rate,
        drop_path_rate=params.drop_path_rate,
        fuse_token=True,
        get_idx=True,
        weight_init=""
    )

    if saved_model_file is None:
        saved_model_file = params.saved_model_fs

    if saved_model_file and os.path.isfile(saved_model_file):
        ckpt = torch.load(saved_model_file, map_location="cpu")

        # --- pick the right state_dict key ---
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "fs_model_state_dict" in ckpt:
                state_dict = ckpt["fs_model_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        # optional: filter only matching shapes
        model_dict = fs_model.state_dict()
        filtered = {k: v for k, v in new_state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}

        missing, unexpected = fs_model.load_state_dict(filtered, strict=False)

        print(f"[LOAD] EViT weights loaded from: {saved_model_file}")
        if missing:
            print(f"[LOAD] Missing keys (not loaded): {missing}")
        if unexpected:
            print(f"[LOAD] Unexpected keys (ignored): {unexpected}")
    else:
        print("[LOAD] No valid checkpoint found — initializing from scratch.")

    return fs_model


# Ft model loading function.
def load_ft_model(arch='r3d', saved_model_file=None, num_classes=400, kin_pretrained=False):
    if arch == 'i3d':
        ft_model = build_i3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'largei3d':
        ft_model = build_largei3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'mvitv2':
        ft_model = wrapper_mvit(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'r3d_18':
        ft_model = wrapper_r3d_18(num_classes=num_classes, pretrained=kin_pretrained)
    else:
        print(f'Architecture {arch} invalid for ft_model. Try \'i3d\', \'largei3d\', \'mvitv2\', or \'r3d_18\'.')
        return
    # Load in saved model.
    if saved_model_file:
        saved_dict = torch.load(saved_model_file)
        try:
            ft_model.load_state_dict(saved_dict['ft_model_state_dict'], strict=True)
        except:
            try:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in saved_dict['ft_model_state_dict'].items():
                    name = k[7:]  # Remove 'module.'
                    new_state_dict[name] = v
                ft_model.load_state_dict(new_state_dict, strict=True)
            except:
                ft_model.i3d.load_state_dict(saved_dict['ft_model_state_dict'], strict=True)

        print(f'ft_model loaded from {saved_model_file} successfully!')
    else:
        print(f'ft_model freshly initialized! Pretrained: {kin_pretrained}')

    return ft_model


# Fb model loading function.
def load_fb_model(arch='r50', saved_model_file=None, num_classes=400, kin_pretrained=False):
    if arch == 'r50':  
        fb_model = build_resnet_predictor(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'i3d':
        fb_model = build_i3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'largei3d':
        fb_model = build_largei3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'mvitv2':
        fb_model = wrapper_mvit(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'r3d_18':
        fb_model = priv_wrapper_r3d_18(num_classes=num_classes, pretrained=kin_pretrained)
    else:
        print(f'Architecture {arch} invalid for fb_model. Try \'r50\'')
        return

    # Load in saved model.
    if saved_model_file:
        saved_dict = torch.load(saved_model_file)
        try:
            fb_model.load_state_dict(saved_dict['fb_model_state_dict'], strict=True)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_dict['fb_model_state_dict'].items():
                name = k[7:]  # Remove 'module.'
                new_state_dict[name] = v
            fb_model.load_state_dict(new_state_dict, strict=True)
        print(f'fb_model loaded from {saved_model_file} successfully!')
    else:
        print(f'fb_model freshly initialized! Pretrained: {kin_pretrained}')

    return fb_model



# Build ResNet model for privacy prediction.
def build_resnet_predictor(num_classes=7, pretrained=True):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=None)

    # Replace fc layer to get desired output.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)

    return model


# Build I3D action recognition model.
def build_i3d_classifier(num_classes=400, pretrained=True):
    temp_classes = 0
    if pretrained:
        temp_classes = num_classes
        num_classes = 400
    model = InceptionI3d(num_classes=num_classes, dropout_keep_prob=0.5)
    if pretrained:
        saved_weights = torch.load(os.path.join('..', 'saved_models', 'rgb_imagenet.pt'))
        model.load_state_dict(saved_weights, strict=True)
    if pretrained and temp_classes != 400:
        model.replace_logits(temp_classes)
    return model

# Build large I3D action recognition model.
def build_largei3d_classifier(num_classes=400, pretrained=True):
    temp_classes = 0
    if pretrained:
        temp_classes = num_classes
        num_classes = 400
    model = wrapper_i3d(num_classes=num_classes)
    if pretrained:
        #saved_weights = torch.load(os.path.join('..', 'saved_models', 'i3d_r50_kinetics.pth'))
        saved_weights = torch.load(os.path.join('/home/create.aau.dk/kp07cm/Singularity_Image/Minimization_Noise/model_weights', 'i3d_r50_kinetics.pth'))
        print(f'i3d_r50_kinetics.pth loaded for ft')
        model.i3d.load_state_dict(saved_weights, strict=True)
    if pretrained and temp_classes != 400:
        model.i3d.fc = nn.Linear(512 * 4, temp_classes)
    return model


# Wrapper to return mlp features and prediction.
class wrapper_r3d_18(nn.Module):
    
    def __init__(self, num_classes=400, pretrained=True):
        super(wrapper_r3d_18, self).__init__()
        self.backbone = r3d_18(weights='DEFAULT' if pretrained else None)
        self.fc = self.backbone.fc
        if num_classes != 400:
            self.fc = nn.Linear(512, num_classes)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feature = self.backbone(x)
        pred = self.fc(feature)
        return pred, feature
    
    
# Wrapper to return mlp prediction for fb.
class priv_wrapper_r3d_18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(priv_wrapper_r3d_18, self).__init__()
        self.backbone = r3d_18(weights='DEFAULT' if pretrained else None)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        return logits


# Wrapper to return mlp features and prediction.
class wrapper_mvit(nn.Module):

    def __init__(self, num_classes=400, pretrained=True):
        super(wrapper_mvit, self).__init__()
        self.backbone = mvit_v2_s(weights='DEFAULT' if pretrained else None)
        self.head = self.backbone.head
        if num_classes != 400:
            self.head[1] = nn.Linear(768, num_classes)
        self.backbone.head = nn.Identity()


    def forward(self, x):
        feature = self.backbone(x)
        pred = self.head(feature)
        return pred, feature


# Feature mlp for stable distinctiveness embedding.
class mlp(nn.Module):

    def __init__(self, final_embedding_size = 128, use_normalization = True):    
        super(mlp, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(2048,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):
        with autocast():
            x = self.relu(self.bn1(self.fc1(x)))
            x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
            return x


# Wrapper to return mlp features and prediction.
class wrapper_i3d(nn.Module):

    def __init__(self, num_classes=102):
        super(wrapper_i3d, self).__init__()
        self.i3d = I3Res50(num_classes=num_classes, use_nl=False)
        self.mlp = mlp()

    def forward(self, x):
        pred, feature = self.i3d(x)
        feature = self.mlp(feature)
        return pred, feature


if __name__ == '__main__':
    inputs = torch.rand((2, 3, 8, 224, 224))
    model = load_ft_model(arch='largei3d', num_classes=102, kin_pretrained=True)

    print(model)
    with torch.no_grad():
        output, feat = model(inputs)
    
    print(f'Output shape is: {output.shape}')
    print(f'Feature shape is: {feat.shape}')
