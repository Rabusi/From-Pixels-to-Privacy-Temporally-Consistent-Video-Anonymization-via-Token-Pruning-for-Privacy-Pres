import argparse
import importlib
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom modules
from aux_code import config as cfg
from aux_code.hmdb51_dl import *
# from aux_code.ucf101_dl_old import *
from Evit_new import EViT
# from Evit import EViT
from visualization import *




####------- GRL lambda schedule -------###
def grl_lambda_schedule(epoch, total_epochs, max_lambda=1.0):
    p = epoch / float(total_epochs)
    return max_lambda * (2. / (1. + np.exp(-10 * p)) - 1.)

#####----------------------------------------------------####


####------- Set random seed for reproducibility -------###
def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Deterministic mode enabled with seed {seed}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] Benchmark mode enabled with seed {seed}")

#####----------------------------------------------------####


#####------------- Optimizer and Scheduler---------------####
def get_optimizer_and_scheduler(model, num_epochs, steps_per_epoch, base_lr=None, weight_decay=None, warmup_epochs=3):
    lr = 1e-5 if base_lr is None else base_lr
    wd = 1e-3 if weight_decay is None else weight_decay

    # === no weight decay on norms/biases ===
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias") or "norm" in n.lower() or "ln" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )

    total_steps  = max(1, num_epochs * steps_per_epoch)
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))

    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[warmup_steps])

    print(f"[INFO] AdamW lr={lr:.6g}  wd(decay)={wd}  wd(no_decay)=0.0")
    print(f"[INFO] Scheduler: warmup {warmup_steps} steps -> cosine {total_steps - warmup_steps} steps (total {total_steps})")

    return optimizer, scheduler

#####----------------------------------------------------####


####-----------------Train Epoch------------------------#####

def train_epoch(epoch,model,train_dataloader,optimizer,scheduler,writer,use_cuda,device_name,params,lambda_grl):
    
    model.train()
    scaler = GradScaler()

    Loss_act, Loss_priv, Total_loss = [], [], []
    Correct, Total = 0, 0

    return_masked_video = bool(getattr(params, "return_masked_video_train", False))

    mask_video_to_log = None
    x_masked_to_log = None

    for i, (data, y_act, y_priv, _, _) in enumerate(train_dataloader):
        if use_cuda:
            data = data.to(device_name, non_blocking=True)
            y_act = y_act.to(device_name, non_blocking=True)
            y_priv = y_priv.to(device_name, non_blocking=True).float().squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            if return_masked_video:
                act_logits, priv_logits, idxs_global, mask_video, x_masked = model(data,keep_rate=params.keep_rate,get_idx=True,lambda_grl=lambda_grl,return_masked_video=True)
            else:
                act_logits, priv_logits, idxs_global, _, _ = model(data,keep_rate=params.keep_rate,get_idx=True,lambda_grl=lambda_grl,return_masked_video=False)
                mask_video, x_masked = None, None

            L_act = F.cross_entropy(act_logits, y_act)
            L_priv = F.binary_cross_entropy_with_logits(priv_logits, y_priv)
            loss = params.lambda_act * L_act + L_priv

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        preds = act_logits.argmax(dim=1)
        Correct += (preds == y_act).sum().item()
        Total += y_act.size(0)

        Loss_act.append(float(L_act.detach().cpu()))
        Loss_priv.append(float(L_priv.detach().cpu()))
        Total_loss.append(float(loss.detach().cpu()))

        if return_masked_video and i == 0:
            mask_video_to_log = mask_video.detach().cpu()
            x_masked_to_log = x_masked.detach().cpu()

        if i % 200 == 0:
            print(
                f"Epoch:{epoch}, Batch {i}, "
                f"act_acc: {100 * Correct / max(Total,1):.2f}%, "
                f"L_act: {L_act.item():.4f}, "
                f"L_priv: {L_priv.item():.4f}, "
                f"total_loss: {loss.item():.4f}",
                flush=True,
            )

    # ---- Logging ----
    act_acc = Correct / max(Total, 1)
    loss_act = float(np.mean(Loss_act)) if len(Loss_act) else 0.0
    loss_priv = float(np.mean(Loss_priv)) if len(Loss_priv) else 0.0
    train_loss = float(np.mean(Total_loss)) if len(Total_loss) else 0.0

    if writer is not None:
        writer.add_scalar("Action/Acc", act_acc, epoch)
        writer.add_scalar("Train/Loss_act", loss_act, epoch)
        writer.add_scalar("Train/Loss_priv", loss_priv, epoch)
        writer.add_scalar("Train/Total_loss", train_loss, epoch)
        writer.flush()

        if idxs_global and len(idxs_global) > 0:
            kept_last = (idxs_global[-1] >= 0).sum(dim=1).float().mean().item()
            writer.add_scalar("Pruning/KeptTubeletsLast", kept_last, epoch)

    print(f"Epoch {epoch}: act_acc={act_acc*100:.2f}%, "
          f"L_act={loss_act:.4f}, L_priv={loss_priv:.4f}, total_loss={train_loss:.4f}",
          flush=True)

    return act_acc, loss_act, loss_priv, train_loss, mask_video_to_log, x_masked_to_log

#######------------------------------------------------------######



#####------- Visualization of anonymized video frames -------#####

#####------- Visualization of anonymized video frames -------#####

def _to_uint8_frames(x_cthw):
    """
    x_cthw: (C,T,H,W) float tensor
    clamps to [0,1] then returns (T,H,W,3) uint8
    """
    x = x_cthw.detach().float().cpu().clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).contiguous().numpy()  # (T,H,W,3)


def visualize_pruned_video(
    model,
    val_loader,
    device,
    out_dir,
    keep_rate,
    n_samples=5,
    n_frames=10,
    overlay=True,
):
    """
    Saves ONE PNG per sample:
      - rows: Original / Masked / Overlay (if overlay=True)
      - cols: n_frames (default 10), evenly spaced across T

    Requires model forward to return:
      act_logits, priv_logits, idxs_global, mask_video, x_masked
    when called with get_idx=True and return_masked_video=True.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    saved = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device, non_blocking=True)
            y_act = batch[1] if len(batch) > 1 else None

            act_logits, priv_logits, idxs_global, mask_video, x_masked = model(
                data,
                keep_rate=keep_rate,
                get_idx=True,
                return_masked_video=True,
            )

            data_cpu = data.detach().cpu()
            x_masked_cpu = x_masked.detach().cpu()
            mask_cpu = mask_video.detach().cpu()

            B, C, T, H, W = data_cpu.shape

            # pick 10 evenly spaced frames
            if T >= n_frames:
                frame_ids = np.linspace(0, T - 1, n_frames).round().astype(int).tolist()
            else:
                frame_ids = list(range(T)) + [T - 1] * (n_frames - T)

            for b in range(B):
                if saved >= n_samples:
                    return

                orig = _to_uint8_frames(data_cpu[b])         # (T,H,W,3)
                masked = _to_uint8_frames(x_masked_cpu[b])   # (T,H,W,3)
                m = mask_cpu[b, 0].cpu().numpy()             # (T,H,W) 0/1

                pred = int(act_logits[b].argmax().item())
                gt = int(y_act[b].item()) if y_act is not None else None

                kept_cnt = None
                if idxs_global and len(idxs_global) > 0:
                    kept_cnt = int((idxs_global[-1][b] >= 0).sum().item())  # ignore fused -1

                rows = 3 if overlay else 2
                fig, axes = plt.subplots(rows, n_frames, figsize=(1.6 * n_frames, 1.8 * rows))
                axes = np.array(axes).reshape(rows, n_frames)

                for j, t in enumerate(frame_ids):
                    axes[0, j].imshow(orig[t])
                    axes[0, j].axis("off")

                    axes[1, j].imshow(masked[t])
                    axes[1, j].axis("off")

                    if overlay:
                        axes[2, j].imshow(orig[t])
                        axes[2, j].imshow(m[t], alpha=0.35)
                        axes[2, j].axis("off")

                title = f"sample{saved}_pred{pred}"
                if gt is not None:
                    title += f"_gt{gt}"
                if kept_cnt is not None:
                    title += f"_kept{kept_cnt}"

                fig.suptitle(title, fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{title}_grid{n_frames}.png"), dpi=160)
                plt.close(fig)

                saved += 1
######------------------------------------------------------######

######------------------------------------------------------######

    
#####---------------------- Validation ----------------------#####

def val_epoch(epoch, mode, cropping_fac, act_pred_dict, act_label_dict, priv_pred_dict, priv_label_dict, val_loader, model, use_cuda, device_name, params):
    print(f'Validation at epoch {epoch}, mode={mode}, crop={cropping_fac}.')
    model.eval()

    act_losses, priv_losses = [], []
    act_preds, act_gts = [], []
    priv_preds, priv_gts = [], []
    vid_paths = []

    with torch.no_grad():
        for i, (data, y_act, y_priv, vid_path, _) in enumerate(val_loader):
            if use_cuda:
                data = data.to(device_name, non_blocking=True)
                y_act = y_act.to(device_name, non_blocking=True).long()
                y_priv = y_priv.to(device_name, non_blocking=True).float().squeeze(1)

            act_logits, priv_logits, _, _, _ = model(data, keep_rate=params.keep_rate, get_idx=True, lambda_grl=0.0)

            act_loss = F.cross_entropy(act_logits, y_act)
            priv_loss = F.binary_cross_entropy_with_logits(priv_logits, y_priv)

            act_losses.append(act_loss.item())
            priv_losses.append(priv_loss.item())

            act_preds.append(act_logits.detach().cpu().numpy())
            act_gts.append(y_act.detach().cpu().numpy())
            priv_preds.append(torch.sigmoid(priv_logits).detach().cpu().numpy())
            priv_gts.append(y_priv.detach().cpu().numpy())
            vid_paths.extend(vid_path)

            if i % 100 == 0:
                print(f'Validation Epoch {epoch}, Batch {i}, '
                      f'Val Loss Action : {np.mean(act_losses):.4f}, '
                      f'Val Loss Privacy: {np.mean(priv_losses):.6f}', flush=True)

    # Convert to numpy
    act_preds = np.concatenate(act_preds, axis=0)
    act_gts = np.concatenate(act_gts, axis=0)
    priv_preds = np.concatenate(priv_preds, axis=0)
    priv_gts = np.concatenate(priv_gts, axis=0)

    # --- Action accuracy ---
    c_pred = np.argmax(act_preds, axis=1)
    correct = np.sum(c_pred == act_gts)
    act_acc = correct / len(c_pred)
    print(f'Correct action predictions: {correct} / {len(c_pred)}', flush=True)

    for vp, pred, gt in zip(vid_paths, act_preds, act_gts):
        fname = os.path.basename(vp)
        act_pred_dict.setdefault(fname, []).append(pred)
        act_label_dict.setdefault(fname, gt)

    # --- Privacy metrics ---
    y_bin = (priv_preds > 0.5).astype(int)
    prec, recall, f1, _ = precision_recall_fscore_support(priv_gts, y_bin, average=None, zero_division=0)
    ap = average_precision_score(priv_gts, priv_preds, average=None)

    print(f'Macro f1: {np.mean(f1):.4f}, prec: {np.mean(prec):.4f}, recall: {np.mean(recall):.4f}', flush=True)
    print(f'Classwise AP: {ap}', flush=True)
    print(f'Macro AP: {np.mean(ap):.4f}', flush=True)

    for vp, pred, gt in zip(vid_paths, priv_preds, priv_gts):
        fname = os.path.basename(vp)
        priv_pred_dict.setdefault(fname, []).append(pred)
        priv_label_dict.setdefault(fname, gt)

    return (act_pred_dict, act_label_dict, act_acc, np.mean(act_losses), priv_pred_dict, priv_label_dict, np.mean(ap))

######------------------------------------------------------######


    
######----------------- Main training script -----------------######

def train_model(params, devices):
    # print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    
    torch.cuda.empty_cache()
    
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
    def model_summary(model, img_size, patch_size, all_frames=16, tubelet_size=2):
        """
        Prints a concise summary of the EViT model.
        """
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_patches_per_frame = (img_size // patch_size) ** 2
        num_tubelets = all_frames // tubelet_size
        total_patches = num_patches_per_frame * num_tubelets

        print("\n================= EViT Model Summary =================")
        print(f"Total parameters      : {num_params:,}")
        print(f"Trainable parameters  : {num_trainable:,}")
        print(f"Embedding dimension   : {model.embed_dim}")
        print(f"Transformer depth     : {model.depth}")
        print(f"Number of heads       : {model.num_heads}")
        print(f"Patches per frame     : {num_patches_per_frame}")
        print(f"Tubelets per clip     : {num_tubelets}")
        print(f"Total visual tokens   : {total_patches}")
        print(f"Dropout rate          : {model.drop_rate}")
        print(f"Attention dropout     : {model.blocks[0].attn.attn_drop.p}")
        print(f"DropPath max rate     : {model.blocks[-1].drop_path.drop_prob}")
        print(f"fuse_token=True, weight_init='imagenet backbone'")
        print("=======================================================\n")

    # 1) Initialize EViT model
    model = EViT(
        img_size = params.img_size,
        patch_size = params.patch_size,
        in_chans = params.in_chans,
        num_classes = params.num_action,
        num_priv = params.num_priv,
        embed_dim = params.embed_dim,
        depth = params.depth,
        num_heads = params.num_heads,
        mlp_ratio = params.mlp_ratio,
        keep_rate = params.keep_rate,
        drop_rate = params.drop_rate,              
        attn_drop_rate = params.attn_drop_rate,    
        drop_path_rate = params.drop_path_rate,    
        fuse_token = True,
        get_idx = True,
        weight_init = ""                   
    )
    
    # Print model summary
    model_summary(model, params.img_size, params.patch_size, params.all_frames, params.tubelet_size)

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUs found!')
        model = nn.DataParallel(model, device_ids=devices)
        model.cuda()
    else:
        print('Only 1 GPU is available')
        model.to(device=torch.device(device_name))

    # 2) Check for checkpoint
    checkpoint_path = os.path.join(save_dir, 'model_21.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch1 = checkpoint['epoch']
        print(f"Resuming training from epoch {epoch1}")

    else:
        print("No checkpoint found, starting from scratch.")
        epoch1 = 1

        # 3) load pretrained backbone 
        if hasattr(cfg, "pretrained_vit_path") and os.path.exists(cfg.pretrained_vit_path):
            print(f"Initializing from pretrained ViT backbone → {cfg.pretrained_vit_path}")
            checkpoint = torch.load(cfg.pretrained_vit_path, map_location='cpu')
            
            # Handle nested dict formats
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
            model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model

            # --- Expand patch embedding (temporal kernel) ---
            if "patch_embed.proj.weight" in state_dict:
                pretrained_w = state_dict["patch_embed.proj.weight"]  
                target_w = model_to_use.patch_embed.proj.weight        
                
                if pretrained_w.shape != target_w.shape:
                    print(f"Adapting patch_embed.proj.weight from {pretrained_w.shape} -> {target_w.shape}")
                    state_dict["patch_embed.proj.weight"] = pretrained_w.repeat(1, 1, target_w.shape[2], 1, 1) / target_w.shape[2]

            # --- Interpolate positional embedding (tokens) ---
            if "pos_embed" in state_dict:
                pos_embed_pretrained = state_dict["pos_embed"]            
                pos_embed_model = model_to_use.pos_embed                  

                if pos_embed_pretrained.shape != pos_embed_model.shape:
                    print(f"Interpolating pos_embed from {pos_embed_pretrained.shape} -> {pos_embed_model.shape}")
                    
                    num_extra_tokens = 1
                    cls_pos = pos_embed_pretrained[:, :num_extra_tokens]
                    spatial_pos = pos_embed_pretrained[:, num_extra_tokens:]
                    hw = int(spatial_pos.shape[1] ** 0.5)
                    
                    spatial_pos = spatial_pos.reshape(1, hw, hw, -1).permute(0, 3, 1, 2)
                    new_hw = int(((pos_embed_model.shape[1] - model_to_use.num_tokens) / model_to_use.num_tubelet) ** 0.5)
                    spatial_pos = F.interpolate(spatial_pos, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
                    spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
                    spatial_pos = spatial_pos.repeat(1, model_to_use.num_tubelet, 1)
                    
                    new_pos = torch.cat([cls_pos.repeat(1, model_to_use.num_tokens, 1), spatial_pos], dim=1)
                    state_dict["pos_embed"] = new_pos

            # --- Load the modified weights ---
            missing, unexpected = model_to_use.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained backbone — Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            if len(missing) > 0:
                print("Missing parameters (not found in pretrained weights):")
                for k in missing:
                    print("  ", k)
            if len(unexpected) > 0:
                print("Unexpected parameters (in checkpoint but not in model):")
                for k in unexpected:
                    print("  ", k)

            # Rough parameter coverage summary
            n_loaded = sum(p.numel() for n, p in model_to_use.named_parameters() if n not in missing and 'head' not in n)
            n_total = sum(p.numel() for p in model_to_use.parameters())
            print(f"Roughly {n_loaded / n_total * 100:.1f}% of parameters initialized from pretrained weights")

            # --- Reinitialize task heads ---
            trunc_normal_(model_to_use.head_act.weight, std=0.02)
            nn.init.zeros_(model_to_use.head_act.bias)

            trunc_normal_(model_to_use.head_priv.weight, std=0.02)
            nn.init.zeros_(model_to_use.head_priv.bias)

        else:
            print("No pretrained weights found — using random initialization.")
    
    # 4) load datasets
    
    def worker_init_fn(worker_id):
        """
        Ensures dataloader workers have deterministic behavior.
        """
        seed = 42 + worker_id
        np.random.seed(seed)
        random.seed(seed)
    
    action_name = cfg.hmdb51_class_mapping
    
    # train_dataset = vpucf_train_dataloader(params, shuffle=True, data_percentage=params.data_percentage_vpucf) 
    train_dataset = vphmdb_train_dataloader(params, action_name=action_name, shuffle=True, data_percentage=params.data_percentage)   
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=False,
                                  batch_size=params.batch_size, 
                                  collate_fn=collate_fn_train,
                                  num_workers=params.num_workers,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True, drop_last = True)

    
    print(f'Train dataset length: {len(train_dataset)}')
    steps_per_epoch = len(train_dataloader)
    print(f'Train dataset steps per epoch: {steps_per_epoch}')
    total_steps = steps_per_epoch * params.num_epochs
    print(f'Total training steps: {total_steps}')
    
    optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs=params.num_epochs,               
                                                       steps_per_epoch=steps_per_epoch,            
                                                       base_lr=params.learning_rate,                
                                                       weight_decay=params.weight_decay,            
                                                       warmup_epochs=5)      

    
    val_array = [1, 5, 10, 12, 15, 20, 25, 30, 35] + [40 + x*2 for x in range(30)]

    best_acc = 0
    best_macro_ap = 1.0
    best_acc_for_privacy = 0
    best_privacy_ap = 1.0

    # 5) # ---------------- TRAINING LOOP ----------------
    for epoch in range(epoch1, params.num_epochs + 1):
        start = time.time()
        
        lambda_grl = grl_lambda_schedule(epoch, params.num_epochs+1, max_lambda=1.0)
        print(f"\nEpoch {epoch} started  |  λ_GRL = {lambda_grl:.4f}")
        
        # ---- Training ----
        act_acc, loss_act, loss_priv, train_loss, mask_video_to_log, x_masked_to_log = train_epoch(epoch, model, train_dataloader, 
                                                                                                   optimizer, scheduler, writer, use_cuda, 
                                                                                                   device_name, params, lambda_grl)
        
        # ---- Validation ----
        if epoch in val_array:
            act_pred_dict, act_label_dict = {}, {}
            priv_pred_dict, priv_label_dict = {}, {}
            act_val_losses, act_mode_accs = [], []
            priv_macro_aps = []

            hflips = params.hflip
            cropping_facs = params.cropping_facs
            modes = list(range(params.num_modes))


            for mode in modes:
                for cropping_fac in [cropping_facs[0]]:
                    for hflip in hflips:
                        validation_dataset = vphmdb_validation_dataloader(params=params,
                                                                          action_name=action_name,         
                                                                          shuffle=True, 
                                                                          data_percentage=1.0, 
                                                                          split=1, mode=mode, 
                                                                          hflip=hflip, 
                                                                          cropping_factor=cropping_fac, 
                                                                          threeCrops=False)

                        validation_dataloader = DataLoader(validation_dataset, 
                                                       batch_size=params.batch_size, 
                                                       shuffle=False, 
                                                       num_workers=params.num_workers,
                                                       worker_init_fn=worker_init_fn, 
                                                       pin_memory=True, 
                                                       collate_fn=collate_fn_val, 
                                                       drop_last=False)

                        if mode == 0 and cropping_fac == cropping_facs[0] and hflip == 0:
                            print(f'Validation set: {len(validation_dataset)} videos | Steps: {len(validation_dataset) / params.batch_size:.2f}')

                        act_pred_dict, act_label_dict, act_accuracy, act_losses, priv_pred_dict, priv_label_dict, macro_ap = val_epoch(epoch, mode, cropping_fac, act_pred_dict, 
                                                                                                                                       act_label_dict, priv_pred_dict, 
                                                                                                                                       priv_label_dict, validation_dataloader, 
                                                                                                                                       model, use_cuda, device_name, params)

                        act_val_losses.append(act_losses)
                        act_mode_accs.append(act_accuracy)
                        priv_macro_aps.append(macro_ap)
                        print(f'Validation at: mode={mode}, crop={cropping_fac}, flip={hflip} → Acc: {act_accuracy * 100:.2f}%, , Macro AP: {macro_ap:.4f}', flush=True)
                        
                        output_dir = os.path.join(cfg.vis_dir_video, params.run_id, f"epoch_{epoch}_mode_{mode}")
                        os.makedirs(output_dir, exist_ok=True)

                        visualize_pruned_video(model=model, val_loader=validation_dataloader,
                                               device=device_name, out_dir=output_dir,
                                               keep_rate=params.keep_rate_eval, n_samples=10,
                                               n_frames=10, overlay=True)
                        

            # ---- Aggregate across all crops/flips/modes ----
            act_predictions = np.zeros((len(act_pred_dict), params.num_action))
            act_ground_truth = []
            for entry, key in enumerate(act_pred_dict.keys()):
                act_predictions[entry] = np.mean(act_pred_dict[key], axis=0)
            for key in act_label_dict.keys():
                act_ground_truth.append(act_label_dict[key])

            act_pred_array = np.flip(np.argsort(act_predictions, axis=1), axis=1)
            c_pred = act_pred_array[:, 0]
            correct_count = np.sum(c_pred == act_ground_truth)
            act_accuracy = float(correct_count) / len(c_pred)
            act_val_loss = np.mean(act_val_losses)
            
            #privacy aggregation
            final_macro_ap = np.mean(priv_macro_aps)
            
            cm = confusion_matrix(act_ground_truth, c_pred)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix at Epoch {epoch}')
            plt.colorbar()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            vis_dir = os.path.join(cfg.vis_dir_cm, params.run_id)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            cm_path = os.path.join(vis_dir, f'confusion_matrix_epoch_{epoch}.png')
            plt.savefig(cm_path)
            plt.close()

            print("\n===== Validation Summary =====")
            print(f"Epoch {epoch} completed with validation Loss: {act_val_loss:.4f}")
            print(f"Action Accuracy: {act_accuracy * 100:.3f}% , Correct Action Prediction: {correct_count}/{len(c_pred)}), Macro AP: {final_macro_ap:.4f}", flush=True)
            print("==============================\n")

            # ---- TensorBoard logging ----
            writer.add_scalar('Validation Loss', act_val_loss, epoch)
            writer.add_scalar('Validation Accuracy', act_accuracy, epoch)
            writer.add_scalar('Validation Macro AP', final_macro_ap, epoch)
            writer.flush()
            
            # ---- Save best model for action accuracy ----
            if (act_accuracy > best_acc) or ((act_accuracy == best_acc) and (final_macro_ap < best_macro_ap)):
                print("++++++++++++++++++++++++++++++")
                print(f"[BEST ACTION MODEL] Epoch {epoch}: Accuracy {act_accuracy*100:.2f}%, Macro AP {final_macro_ap:.4f}")
                print("++++++++++++++++++++++++++++++")

                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"model_best_Action_epoch{epoch}_acc_{act_accuracy:.4f}_macro_ap_{final_macro_ap:.4f}.pth")

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, save_path)

                best_acc = act_accuracy
                best_macro_ap = final_macro_ap

            # ---- Save best model for privacy (lowest macro AP) ----
            if (final_macro_ap < best_privacy_ap) or ((final_macro_ap == best_privacy_ap) and (act_accuracy > best_acc_for_privacy)):
                print("--------------------------------")
                print(f"[BEST PRIVACY MODEL] Epoch {epoch}: Macro AP {final_macro_ap:.4f}, Accuracy {act_accuracy*100:.2f}%")
                print("--------------------------------")

                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"model_best_Privacy_epoch{epoch}_ap_{final_macro_ap:.4f}_act_acc_{act_accuracy*100:.2f}.pth")

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, save_path)

                best_privacy_ap = macro_ap
                best_acc_for_privacy = act_accuracy

        # ---- Save temporary checkpoint ----
        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
        os.makedirs(save_dir, exist_ok=True)
        temp_path = os.path.join(save_dir, 'model_temp.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, temp_path)

        # ---- Timing ----
        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch}: {taken:.2f}s\n')
        print()
        
    writer.close()
    torch.cuda.empty_cache()     
    
######------------------------------------------------------######

   
######------------------main--------------------------------######        
if __name__ == '__main__':
    set_seed(seed=42, deterministic=True)
    parser = argparse.ArgumentParser(description='Script to initialize the model.')

    parser.add_argument("--params_Evit", dest='params', type=str, required=False, default='params_Evit.py', help='params')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None, help='devices should be a list')
    args = parser.parse_args()
    
    #relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(script_dir, args.params)
    
    if os.path.exists(params_file):
        spec = importlib.util.spec_from_file_location("params_Evit", params_file)
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        print(f'{params_file} is loaded as parameter file.')
    else:
        print(f'{params_file} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_model(params, args.devices)