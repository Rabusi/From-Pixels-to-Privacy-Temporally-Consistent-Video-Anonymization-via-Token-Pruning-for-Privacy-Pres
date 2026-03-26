import argparse
import importlib
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aux_code.config as cfg
from aux_code.model_loaders import load_fs_model, load_ft_model
# from aux_code.ucf101_dl_old import *
from aux_code.hmdb51_dl import *


# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True
    


##########---------------------- Training epoch -----------------##########

def train_epoch(epoch, data_loader, fs_model, ft_model, criterion,
                optimizer, scheduler, writer, use_cuda, lr, scaler, device_name, params):
    print(f"Train at epoch {epoch}")

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    if writer is not None:
        writer.add_scalar("Learning Rate", lr, epoch)
    print(f"Learning rate is: {optimizer.param_groups[0]['lr']}")

    losses = []
    predictions, gt = [], []

    fs_model.eval()
    ft_model.train()

    device = torch.device(device_name)

    for i, (data, y_act, _, _, _) in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            data = data.to(device=device, non_blocking=True)
            y_act = y_act.to(device=device, non_blocking=True)
        if data.dim() == 5 and data.shape[1] != 3 and data.shape[2] == 3:
            data = data.permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            _, _, _, _, x_masked = fs_model(data, keep_rate=params.keep_rate_eval, get_idx=True, return_masked_video=True, lambda_grl=1.0)

        with autocast():
            out = ft_model(x_masked)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.dim() == 2:
                assert y_act.max().item() < logits.shape[1], (f"Label out of range: y_act.max={y_act.max().item()} >= num_classes={logits.shape[1]}")

            loss = criterion(logits, y_act)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and getattr(params, "scheduler_per_iter", True):
            scheduler.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.detach().cpu().numpy())
        gt.extend(y_act.detach().cpu().numpy())

        if i % 200 == 0:
            print(f"Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}", flush=True)

    if scheduler is not None and not getattr(params, "scheduler_per_iter", True):
        scheduler.step()

    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    if writer is not None:
        writer.add_scalar("Training Loss", mean_loss, epoch)

    predictions = np.asarray(predictions)
    gt = np.asarray(gt)
    accuracy = float((predictions == gt).sum()) / max(1, predictions.size)

    print(f"Training Epoch: {epoch}, Loss: {mean_loss:.5f}")
    print(f"Training Accuracy at Epoch {epoch} is {accuracy*100:.3f}%")

    return ft_model, mean_loss, scaler


##########---------------------------------------------------------##########        


##########---------------------- Validation epoch -----------------##########


def val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, data_loader, fs_model, ft_model, criterion, use_cuda, device_name, params):
    print(f'Validation at epoch {epoch}.')

    fs_model.eval()
    ft_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    device = torch.device(device_name)

    for i, (data, y_act, _, vid_path, _) in enumerate(data_loader):
        vid_paths.extend(list(vid_path))
        ground_truth.extend(y_act.detach().cpu().numpy() if torch.is_tensor(y_act) else y_act)

        if use_cuda:
            data = data.to(device=device, non_blocking=True)
            y_act = y_act.to(device=device, non_blocking=True)

        if data.dim() == 5 and data.shape[1] != 3 and data.shape[2] == 3:
            data = data.permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            _, _, _, _, x_masked = fs_model(data, keep_rate=params.keep_rate_eval, get_idx=True, return_masked_video=True, lambda_grl=1.0)

            out = ft_model(x_masked)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = criterion(logits, y_act)
            losses.append(loss.item())

            probs = nn.functional.softmax(logits, dim=1)
            predictions.extend(probs.detach().cpu().numpy())

        if i % 200 == 0:
            print(f'Validation Epoch {epoch}, Batch {i}, Loss : {np.mean(losses):.5f}', flush=True)

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
    c_pred = pred_array[:, 0]

    for entry in range(len(vid_paths)):
        key = str(vid_paths[entry].split('/')[-1])
        pred_dict.setdefault(key, []).append(predictions[entry])

    for entry in range(len(vid_paths)):
        key = str(vid_paths[entry].split('/')[-1])
        if key not in label_dict:
            label_dict[key] = int(ground_truth[entry])

    correct_count = np.sum(c_pred == ground_truth)
    accuracy = float(correct_count) / max(1, len(c_pred))

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, float(np.mean(losses))


##########--------------------------------------------------------##########        


##########---------------------- Train classifier -----------------##########

def train_classifier(params, devices):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    # Empty cuda cache.
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Build ft_model. 
    ft_model = load_ft_model(arch=params.arch_ft, num_classes=params.num_action, kin_pretrained=True)
    # Load in fs_model.
    fs_model = load_fs_model(saved_model_file=params.saved_model_fs)

    epoch1 = 1

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUS found!')
        fs_model = nn.DataParallel(fs_model, device_ids=devices)
        fs_model.cuda()
        ft_model = nn.DataParallel(ft_model, device_ids=devices)
        ft_model.cuda()
        criterion.cuda()
    else:
        print('Only 1 GPU is available')
        fs_model.to(device=torch.device(device_name))
        ft_model.to(device=torch.device(device_name))
        criterion.to(device=torch.device(device_name))

    # Select optimizer.
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,T_max=params.num_epochs,eta_min=1e-6)

    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_facs

    val_array = params.val_array

    print(f'Base learning rate {params.learning_rate}')

    accuracy = 0
    best_score = 0
    best_acc = 0
    train_loss = 1000
    learning_rate = params.learning_rate
    scheduler_epoch = 0

    action_name = cfg.hmdb51_class_mapping
    
    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()

        # train_dataset = vpucf_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)
        train_dataset = vphmdb_train_dataloader(params, action_name=action_name, shuffle=True, data_percentage=params.data_percentage)

        if epoch == epoch1:
            print(f'Train dataset length: {len(train_dataset)}')
            print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')

        train_dataloader = DataLoader(
            train_dataset, 
            shuffle=True,
            batch_size=params.batch_size, 
            num_workers=params.num_workers,
            drop_last=True,
            collate_fn=collate_fn_train,
            pin_memory=True)

        ft_model, train_loss, scaler = train_epoch(epoch, train_dataloader, 
                                                   fs_model, ft_model, 
                                                   criterion, optimizer, 
                                                   scheduler, writer, use_cuda, 
                                                   learning_rate, scaler, 
                                                   device_name, params)

        if train_loss < best_score:
            best_score = train_loss


        # Validation epoch.
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    # validation_dataset = vpucf_val_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage, mode=mode)
                    validation_dataset = vphmdb_validation_dataloader(params=params, 
                                                    action_name=action_name,
                                                    shuffle=True, 
                                                    data_percentage=params.data_percentage, 
                                                    split=1, mode=mode, 
                                                    hflip=0, 
                                                    cropping_factor=cropping_fac, 
                                                    threeCrops=False)
                    validation_dataloader = DataLoader(validation_dataset, 
                                                       batch_size=params.batch_size, 
                                                       shuffle=True, 
                                                       num_workers=params.num_workers, 
                                                       drop_last=True, 
                                                       collate_fn=collate_fn_val)
                    if val_iter == 0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.batch_size}')
                    pred_dict, label_dict, accuracy, loss = val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, validation_dataloader, fs_model, ft_model, criterion, use_cuda, device_name, params)
                    val_losses.append(loss)

                    predictions = np.zeros((len(list(pred_dict.keys())), params.num_action))
                    ground_truth = []
                    for entry, key in enumerate(pred_dict.keys()):
                        predictions[entry] = np.mean(pred_dict[key], axis=0)

                    for key in label_dict.keys():
                        ground_truth.append(label_dict[key])

                    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                    c_pred = pred_array[:, 0]

                    correct_count = np.sum(c_pred==ground_truth)
                    accuracy_all = float(correct_count)/len(c_pred)
                    print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]}, is {accuracy_all*100:.3f}%')

            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())), params.num_action))
            ground_truth = []

            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy = float(correct_count)/len(c_pred)
            print(f'Val loss for epoch {epoch} is {val_loss}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', accuracy, epoch)
            print(f'Overall Accuracy is for epoch {epoch} is {accuracy*100:.3f}%')

            if accuracy > best_acc:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(accuracy)[:6]}.pth')
                states = {
                    'epoch': epoch + 1,
                    'amp_scaler': scaler,
                    'ft_model_state_dict': ft_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)
                best_acc = accuracy

        # Temp saving.
        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'amp_scaler' : scaler,
            'ft_model_state_dict': ft_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()

##########--------------------------------------------------##########        


##########---------------------- Main code -----------------##########

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline action')
    parser.add_argument("--params_anonymized_action", type=str, required=False,default="params_anonymized_action.py", help="Path to params file")
    parser.add_argument("--devices", dest="devices", action="append", type=int,required=False, default=None, help="devices should be a list")
    args = parser.parse_args()

    # always resolve relative to this script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(script_dir, args.params_anonymized_action)

    if os.path.exists(params_file):
        spec = importlib.util.spec_from_file_location("params", params_file)
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        print(f"{params_file} is loaded as parameter file.")
    else:
        print(f"{params_file} does not exist, change to valid filename.")
        exit(1)

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_classifier(params, args.devices)
