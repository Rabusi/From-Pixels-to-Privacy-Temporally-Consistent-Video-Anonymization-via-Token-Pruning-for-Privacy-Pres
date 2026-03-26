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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import aux_code.config as cfg
from aux_code.models.vivit_act import ViViT_act
from aux_code.ucf101_dl import *


# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, train_data_loader, ft_model, criterion, optimizer, scheduler, writer, use_cuda, lr, scaler, device_name):
    print(f'Train at epoch {epoch}')

    # Update LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('Learning Rate', lr, epoch)  
    print(f'Learning rate is: {lr}')

    losses = []
    predictions, gt = [], []

    ft_model.train()

    for i, (data, y_act, _, _, _) in enumerate(train_data_loader):
        optimizer.zero_grad(set_to_none=True)
        
        if use_cuda:
            data = data.to(device=torch.device(device_name), non_blocking=True)
            y_act = y_act.to(device=torch.device(device_name), non_blocking=True)

        with autocast(dtype=torch.float16):
            output = ft_model(data)
            loss = criterion(output, y_act)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        predictions.extend(output.argmax(dim=1).cpu().numpy())
        gt.extend(y_act.cpu().numpy())

        if i % 100 == 0:
            print(f"Training Epoch {epoch}, Batch {i}, Avg Loss: {np.mean(losses):.5f}", flush=True)

    avg_loss = np.mean(losses)
    accuracy = (np.asarray(predictions) == np.asarray(gt)).mean()

    print(f"Training Epoch {epoch}: Loss={avg_loss:.5f}, Acc={accuracy*100:.3f}%")
    writer.add_scalar('Training Loss', avg_loss, epoch)
    writer.add_scalar('Training Accuracy', accuracy, epoch)

    return ft_model, avg_loss, scaler


# Validation epoch.
def val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict,data_loader, ft_model, criterion, use_cuda, device_name):
    print(f'Validation at epoch {epoch}.')

    ft_model.eval()
    losses, predictions, ground_truth, vid_paths = [], [], [], []

    with torch.no_grad():
        for i, (data, act_label, _, vid_path, _) in enumerate(data_loader):
            vid_paths.extend(vid_path)
            ground_truth.extend(act_label)

            if use_cuda:
                data = data.to(device=torch.device(device_name), non_blocking=True)
                act_label = act_label.to(device=torch.device(device_name),dtype=torch.long, non_blocking=True)

            output = ft_model(data)
            loss = criterion(output, act_label)
            losses.append(loss.item())

            probs = torch.softmax(output, dim=1)
            predictions.extend(probs.cpu().numpy())

            if i % 200 == 0:
                print(f'Validation Epoch {epoch}, Batch {i}, Avg Loss: {np.mean(losses):.5f}', flush=True)

    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    c_pred = np.argmax(predictions, axis=1)

    # Update prediction dictionary
    for entry, vp in enumerate(vid_paths):
        fname = str(vp.split('/')[-1])
        if fname not in pred_dict:
            pred_dict[fname] = []
        pred_dict[fname].append(predictions[entry])

    # Update label dictionary
    for entry, vp in enumerate(vid_paths):
        fname = str(vp.split('/')[-1])
        if fname not in label_dict:
            label_dict[fname] = ground_truth[entry]

    correct_count = np.sum(c_pred == ground_truth)
    accuracy = correct_count / len(c_pred)

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac}, Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)



# Main code. 
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

    # Load in correct model file.
    ft_model = ViViT_act(num_classes=params.num_action, num_frames=params.all_frames, pretrained=True)
    epoch1 = 1

    # Init loss function.
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUS found!')
        ft_model = nn.DataParallel(ft_model, device_ids=devices)
        ft_model.cuda()
        criterion.cuda()
    else:
        print('Only 1 GPU is available')
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

    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()

        train_dataset = vpucf_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)

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

        ft_model, train_loss, scaler = train_epoch(epoch, train_dataloader, ft_model, criterion, optimizer, scheduler, writer, use_cuda, learning_rate, scaler, device_name)

        if train_loss < best_score:
            best_score = train_loss
            scheduler_epoch = 0
        else:
            scheduler_epoch += 1

        # Validation epoch.
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    validation_dataset = vpucf_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    validation_dataloader = DataLoader(validation_dataset, 
                                                       batch_size=params.v_batch_size, 
                                                       shuffle=True, 
                                                       num_workers=params.num_workers, 
                                                       drop_last=True, 
                                                       collate_fn=collate_fn_val)
                    if val_iter == 0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')
                    pred_dict, label_dict, accuracy, loss = val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, validation_dataloader, ft_model, criterion, use_cuda, device_name)
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
        if params.lr_scheduler != 'cosine' and learning_rate < 1e-12 and epoch > 10:
            print(f'Learning rate is very low now, stopping the training.')
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline action')
    parser.add_argument("--params_action", type=str, required=False, default="params_action.py", help="Path to params file")
    parser.add_argument("--devices", dest="devices", action="append", type=int,required=False, default=None, help="devices should be a list")
    args = parser.parse_args()

    # always resolve relative to this script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(script_dir, args.params_action)

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
