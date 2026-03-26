import argparse
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import traceback
import importlib
import params_privacy as params

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aux_code import config as cfg
from aux_code.model_loaders import load_fb_model, load_fs_model
# from aux_code.ucf101_dl import *
from aux_code.hmdb51_dl import *

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


##########---------------------- Training epoch -----------------##########

def train_epoch(epoch, train_dataloader, fs_model, fb_model, anon, criterion, optimizer, scheduler, writer, use_cuda, learning_rate):
    print(f"Train Epoch {epoch}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    if writer is not None:
        writer.add_scalar("Learning Rate", learning_rate, epoch)
    print(f"Learning rate is: {optimizer.param_groups[0]['lr']}")

    losses = []

    fs_model.eval()
    fb_model.train()

    for i, (inputs, _, priv_label, _, _) in enumerate(train_dataloader):
        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            inputs = inputs.cuda()
        if torch.is_tensor(priv_label):
            priv_label = priv_label.cuda().float()
        else:
            priv_label = torch.tensor(np.asarray(priv_label), device=torch.device("cuda"), dtype=torch.float32)

        if priv_label.dim() == 3 and priv_label.size(1) == 1:
            priv_label = priv_label.squeeze(1)

        if inputs.dim() == 5 and inputs.shape[1] != 3 and inputs.shape[2] == 3:
            inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()

        if anon:
            with torch.no_grad():
                _, _, _, _, x_video = fs_model(inputs, keep_rate=params.keep_rate_eval, get_idx=True, return_masked_video=True, lambda_grl=1.0)
        else:
            x_video = inputs

        B, C, T, H, W = x_video.shape
        x2d = x_video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        frame_logits = fb_model(x2d)
        frame_logits = frame_logits[0] if isinstance(frame_logits, (tuple, list)) else frame_logits

        clip_logits = frame_logits.view(B, T, -1).mean(dim=1)

        loss = criterion(clip_logits, priv_label)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i % 100 == 0:
            print(f"Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}", flush=True)

    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    print(f"Training Epoch: {epoch}, Loss: {mean_loss:.4f}", flush=True)
    if writer is not None:
        writer.add_scalar("Training Loss", mean_loss, epoch)

    return fb_model, mean_loss

##########---------------------------------------------------------##########        


##########---------------------- Validation epoch -----------------##########

def val_epoch(epoch, validation_dataloader, mode, cropping_fac, fs_model, fb_model, anon, criterion, use_cuda, writer):
    fs_model.eval()
    fb_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []
    label_dict, pred_dict = {}, {}

    for i, batch in enumerate(validation_dataloader):
        if batch is None:
            continue
        inputs, _, priv_label, vid_path, _ = batch
        if use_cuda:
            inputs = inputs.cuda()

        if torch.is_tensor(priv_label):
            priv_label = priv_label.cuda().float()
        else:
            priv_label = torch.tensor(np.asarray(priv_label), device=torch.device("cuda"), dtype=torch.float32)

        if priv_label.dim() == 3 and priv_label.size(1) == 1:
            priv_label = priv_label.squeeze(1)

        if inputs.dim() == 5 and inputs.shape[1] != 3 and inputs.shape[2] == 3:
            inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            if anon:
                _, _, _, _, x_video = fs_model(inputs, keep_rate=params.keep_rate_eval, get_idx=True, return_masked_video=True, lambda_grl=1.0)
            else:
                x_video = inputs

            B, C, T, H, W = x_video.shape
            x2d = x_video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

            out = fb_model(x2d)
            frame_logits = out[0] if isinstance(out, (tuple, list)) else out

            clip_logits = frame_logits.view(B, T, -1).mean(dim=1)

            loss = criterion(clip_logits, priv_label)
            losses.append(loss.item())

            clip_probs = torch.sigmoid(clip_logits)
            predictions.extend(clip_probs.detach().cpu().numpy())
            ground_truth.extend(priv_label.detach().cpu().numpy())
            vid_paths.extend(list(vid_path))

        if i % 100 == 0:
            print(f'Validation Epoch {epoch}, Batch {i} - Loss: {np.mean(losses):.5f}', flush=True)

    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    print(f'Validation Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac}, Validation Loss: {mean_loss:.5f}', flush=True)
    if writer is not None:
        writer.add_scalar('Validation Loss', mean_loss, epoch)

    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth,(predictions > 0.5).astype(int),average=None,zero_division=0)
    ap = average_precision_score(ground_truth, predictions, average=None)

    print(f'Macro f1: {np.mean(f1):.4f}, Macro prec: {np.mean(prec):.4f}, Macro recall: {np.mean(recall):.4f}')
    print(f'Classwise AP: {ap}')
    print(f'Macro AP: {np.mean(ap):.4f}')

    for entry in range(len(vid_paths)):
        key = str(vid_paths[entry].split('/')[-1])
        pred_dict.setdefault(key, []).append(predictions[entry])

    for entry in range(len(vid_paths)):
        key = str(vid_paths[entry].split('/')[-1])
        if key not in label_dict:
            label_dict[key] = ground_truth[entry]

    return pred_dict, label_dict, float(np.mean(ap))
    

##########--------------------------------------------------------##########        


##########---------------------- Train classifier -----------------##########

def train_classifier(params, devices):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    use_cuda = True
    writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fs_model = load_fs_model(saved_model_file=params.saved_model_fs)
    # Freeze model weights.
    for param in fs_model.parameters():
        param.requires_grad = False
    
    fb_model = load_fb_model(arch=params.arch, saved_model_file=None, num_classes=params.num_priv, kin_pretrained=True)
    
    epoch1 = 1
    criterion = nn.BCEWithLogitsLoss()
    
    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUs found!')
        fs_model = nn.DataParallel(fs_model, device_ids=devices).cuda()
        fb_model = nn.DataParallel(fb_model, device_ids=devices).cuda()
        criterion = criterion.cuda()
    else:
        print('Only 1 GPU is available')
        fs_model.to(torch.device(device_name))
        fb_model.to(torch.device(device_name))
        criterion.to(torch.device(device_name))
    
    optimizer = torch.optim.AdamW(fb_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,T_max=params.num_epochs,eta_min=1e-6)
    
    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_facs
    val_array = params.val_array
    print(f'Base learning rate {params.learning_rate}')
    
    train_loss_best = 1000
    best_score = 0
    
    action_name = cfg.hmdb51_class_mapping
    
    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()
        
        # train_dataset = vpucf_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)
        train_dataset = vphmdb_train_dataloader(params, action_name=action_name, shuffle=True, data_percentage=params.data_percentage)
        
        if epoch == epoch1:
            print(f'Train dataset length: {len(train_dataset)}')
            print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, 
                                      batch_size=params.batch_size, 
                                      num_workers=params.num_workers,
                                      drop_last=True, 
                                      collate_fn=collate_fn_train, 
                                      pin_memory=True)
        
        fb_model, train_loss = train_epoch(epoch, train_dataloader, fs_model, fb_model, params.anon, criterion, optimizer, scheduler, writer, use_cuda, params.learning_rate)

        if train_loss < train_loss_best:
            train_loss_best = train_loss
            print(f'Best training loss till now: {train_loss_best}')
            
        # Validation epoch
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []
            
            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    # validation_dataset = vpucf_val_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage, mode=mode)
                    validation_dataset = vphmdb_validation_dataloader(params=params, 
                                action_name=action_name,
                                shuffle=True, 
                                data_percentage=0.01, 
                                split=1, mode=mode, 
                                hflip=0, 
                                cropping_factor=cropping_fac, 
                                threeCrops=False)
                    validation_dataloader = DataLoader(validation_dataset, shuffle=True, 
                                                       batch_size=params.batch_size, 
                                                       num_workers=params.num_workers, 
                                                       drop_last=True, collate_fn=collate_fn_val, 
                                                       pin_memory=True)
                    if val_iter == 0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.batch_size}')
                    
                    pred_dict, label_dict, macro_ap = val_epoch(epoch, validation_dataloader, mode, cropping_fac, fs_model, fb_model, params.anon, criterion, use_cuda, writer)
                    
                    if macro_ap > best_score:
                        best_score = macro_ap
                        print('++++++++++++++++++++++++++++++')
                        print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                        print('++++++++++++++++++++++++++++++')
                        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_file_path = os.path.join(save_dir, f'model_{epoch}_loss_{macro_ap:.6f}.pth')
                        states = {
                            'epoch': epoch + 1,
                            'priv_net_state_dict': fb_model.state_dict(),
                            'pred_dict': pred_dict,
                            'label_dict': label_dict,
                            'optimizer': optimizer.state_dict()
                        }
                        torch.save(states, save_file_path)

            save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'priv_net_state_dict': fb_model.state_dict(),
                'pred_dict': pred_dict,
                'label_dict': label_dict,
                'optimizer': optimizer.state_dict()
                }
            torch.save(states, save_file_path)

        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')

##########--------------------------------------------------##########        


##########---------------------- Main code -----------------##########

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline sparsified privacy model.')
    parser.add_argument("--params_privacy", type=str, required=False, default='params_privacy.py', help='path to params_privacy file')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None, help='devices should be a list')
    args = parser.parse_args()
    
    #relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(script_dir, args.params_privacy)
    
    if os.path.exists(params_file):
        spec = importlib.util.spec_from_file_location("params", params_file)
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        print(f'{params_file} is loaded as parameter file.')
    else:
        print(f'{params_file} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_classifier(params, args.devices)

