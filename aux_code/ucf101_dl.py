import os
import ast
import numpy as np
import imageio
from PIL import Image
import json
import random
import torch
import pandas as pd
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
# from decord import VideoReader, cpu

# Decord after torch.
import decord

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import aux_code.config as cfg
from sparsification import params_Evit as params

decord.bridge.set_bridge('torch')

# Training dataloader.
class vpucf_train_dataloader(Dataset):
    def __init__(self, params, shuffle=True, data_percentage=1.0, split=1, frame_wise_aug=False):
        self.params = params
        self.shuffle = shuffle
        self.data_percentage = data_percentage
        self.framewise_aug = frame_wise_aug

        self.img_size = params.img_size
        self.all_frames = params.all_frames
        
        if split <= 3:
            split_file = os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'trainlist0{split}.txt')
            all_paths = open(split_file, 'r').read().splitlines()
            vid_path = [line.strip().split(' ')[0].replace('/', os.sep) for line in all_paths]
            self.all_train_vid_path = [
                os.path.join(cfg.ucf101_path, x.replace('.mp4', '.avi').replace('/', os.sep))
                for x in vid_path
            ]
        else:
            raise ValueError(f"Invalid split number: {split}. Must be 1, 2, or 3.")

        # Load private labels
        priv_df = pd.read_csv(cfg.vp_ucf101_private_label)
        priv_df['Array Values'] = priv_df['Array Values'].apply(ast.literal_eval)
        priv_df['Full Path'] = priv_df['File Path'].apply(
            lambda x: os.path.join(cfg.ucf101_path, x.replace('.mp4', '.avi'))
        )
        
        # Get only rows where the video path is in the training split
        final_vid_path = priv_df[priv_df['Full Path'].isin(self.all_train_vid_path)]
        self.video_paths = final_vid_path['Full Path'].tolist()
        self.priv_labels = final_vid_path['Array Values'].tolist()        

        # Load action class mapping
        self.action_classes = json.load(open(cfg.ucf101_class_mapping))['classes']

        # Shuffle and sample
        if self.shuffle:
            combined = list(zip(self.video_paths, self.priv_labels))
            random.shuffle(combined)
            self.video_paths, self.priv_labels = zip(*combined)
            self.video_paths = list(self.video_paths)
            self.priv_labels = list(self.priv_labels)

        self.data_limit = int(len(self.video_paths) * self.data_percentage)
        self.video_paths = self.video_paths[:self.data_limit]
        self.priv_labels = self.priv_labels[:self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19
        self.framewise_aug = frame_wise_aug

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        clip, act_label, priv_label, vid_path, frame_list = self.process_data(index)

        if clip is None or priv_label is None or act_label is None:
            print(f"[SKIP] {vid_path}")
            return None  # Skip invalid samples

        return clip, act_label, priv_label, vid_path, frame_list
    
    def process_data(self, idx):
        vid_path = self.video_paths[idx]
        priv_label = self.priv_labels[idx]
        act_class_name = os.path.basename(os.path.dirname(vid_path))
        act_label = self.action_classes.get(act_class_name, None)
        if act_label is None:
            raise ValueError(f"Invalid class name '{act_class_name}' not found in class mapping.")
        act_label -= 1
        
        # clip building:
        clip, frame_list = self.build_clip(vid_path)
        
        return clip, act_label, priv_label, vid_path, idx
    
    def build_clip(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            # ---------- robust frame sampling ----------
            T = self.params.all_frames          # number of frames needed
            skip = int(self.params.fix_skip)    # fixed skip between frames
            max_required = (T - 1) * skip

            if frame_count <= T:
                frames_full = np.linspace(0, frame_count - 1, T).astype(int)
            elif frame_count <= max_required:
                skip = max(1, frame_count // T)
                frames_full = np.arange(0, skip * T, skip)
                frames_full = np.clip(frames_full, 0, frame_count - 1)
            else:
                start = np.random.randint(0, frame_count - max_required)
                frames_full = start + np.arange(0, skip * T, skip)

            frames = vr.get_batch(frames_full).permute(0, 3, 1, 2)  # [T, C, H, W]

            self.ori_reso_h, self.ori_reso_w = frames.shape[2], frames.shape[3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            # ---------- random aug params ----------
            random_array = np.random.rand(2, 10)
            x_erase = np.random.randint(0, self.params.reso_w, size=(2,))
            y_erase = np.random.randint(0, self.params.reso_h, size=(2,))
            cropping_factor1 = np.random.uniform(self.params.min_crop_factor_training, 1, size=(2,))

            if not self.params.no_ar_distortion:
                x0 = np.random.randint(0, (self.ori_reso_w - self.ori_reso_w * cropping_factor1[0]) + 1)
                y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h * cropping_factor1[1]) + 1)
            else:
                size = int(self.min_size * cropping_factor1[0])
                x0 = np.random.randint(0, (self.ori_reso_w - size) + 1)
                y0 = np.random.randint(0, (self.ori_reso_h - size) + 1)

            contrast_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(
                int((self.ori_reso_h / 6) * (self.params.reso_h / 224)),
                int((self.ori_reso_h / 3) * (self.params.reso_h / 224)), size=(2,)
            )
            erase_size2 = np.random.randint(
                int((self.ori_reso_w / 6) * (self.params.reso_h / 224)),
                int((self.ori_reso_w / 3) * (self.params.reso_h / 224)), size=(2,)
            )
            random_color_dropped = np.random.randint(0, 3, (2,))

            # ---------- framewise augmentation ----------
            full_clip = []

            for frame in frames:
                if self.framewise_aug:
                    contrast_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
                    hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
                    saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
                    brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
                    gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
                    erase_size1 = np.random.randint(int(self.erase_size / 2), self.erase_size, size=(2,))
                    erase_size2 = np.random.randint(int(self.erase_size / 2), self.erase_size, size=(2,))
                    random_color_dropped = np.random.randint(0, 3, (2,))

                if self.params.weak_aug:
                    aug_frame = self.weak_augmentation(frame, cropping_factor1[0], x0, y0)
                else:
                    aug_frame = self.augmentation(
                        frame, random_array[0], x_erase, y_erase,
                        cropping_factor1[0], x0, y0,
                        contrast_factor1[0], hue_factor1[0], saturation_factor1[0],
                        brightness_factor1[0], gamma1[0],
                        erase_size1, erase_size2, random_color_dropped[0]
                    )

                full_clip.append(aug_frame)

            # ---------- safety: ensure exactly T frames ----------
            if len(full_clip) != T:
                if len(full_clip) == 0:
                    return None, None
                while len(full_clip) < T:
                    full_clip.append(full_clip[-1].clone())
                full_clip = full_clip[:T]

            return full_clip, frames_full.tolist()

        except Exception as e:
            print(f"[build_clip Exception] {vid_path}: {e}")
            return None, None
        
        
    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1, x0, y0, contrast_factor1, hue_factor1, 
                     saturation_factor1, brightness_factor1, gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)

        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) 
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) 
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) 
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1)
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[7] < 0.4 :
            image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        if random_array[8] <0.4 :
            image = trans.functional.erase(image, x_erase[1], y_erase[1], erase_size1[1], erase_size2[1], v=0) 

        return image
    
    def weak_augmentation(self, image, cropping_factor1, x0, y0):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True) 
        
        image = trans.functional.to_tensor(image)

        return image   

  
    
        
    
# Validation dataset.
class vpucf_val_dataloader(Dataset):
    def __init__(self, params, shuffle=True, data_percentage=1.0, split=1, mode=0, hflip=0, cropping_factor=0.8,threeCrops=False):
        self.total_num_modes = params.num_modes
        self.threecrop = threeCrops
        self.params = params
        self.shuffle = shuffle
        self.data_percentage = data_percentage
        self.img_size = params.img_size
        self.all_frames = params.all_frames
        
        if split <= 3:
            split_file = os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist0{split}.txt')
            all_paths = open(split_file, 'r').read().splitlines()
            vid_path = [line.strip().split(' ')[0].replace('/', os.sep) for line in all_paths]
            self.all_test_vid_path = [
                os.path.join(cfg.ucf101_path, x.replace('.mp4', '.avi').replace('/', os.sep))
                for x in vid_path
            ]
        else:
            raise ValueError(f"Invalid split number: {split}. Must be 1, 2, or 3.")
        
        
        # Load private labels
        priv_df = pd.read_csv(cfg.vp_ucf101_private_label)
        priv_df['Array Values'] = priv_df['Array Values'].apply(ast.literal_eval)
        priv_df['Full Path'] = priv_df['File Path'].apply(
            lambda x: os.path.join(cfg.ucf101_path, x.replace('.mp4', '.avi'))
        )
        
        # Get only rows where the video path is in the training split
        final_vid_path = priv_df[priv_df['Full Path'].isin(self.all_test_vid_path)]

        self.video_paths = final_vid_path['Full Path'].tolist()
        self.priv_labels = priv_df['Array Values'].tolist()
        
        # Load action class mapping
        self.action_classes = json.load(open(cfg.ucf101_class_mapping))['classes']

        if self.shuffle:
            combined = list(zip(self.video_paths, self.priv_labels))
            random.shuffle(combined)
            self.video_paths, self.priv_labels = zip(*combined)
            self.video_paths = list(self.video_paths)
            self.priv_labels = list(self.priv_labels)

        self.data_limit = int(len(self.video_paths) * self.data_percentage)
        self.video_paths = self.video_paths[:self.data_limit]
        self.priv_labels = self.priv_labels[:self.data_limit]
        self.PIL = trans.ToPILImage()
        self.Tensor = trans.ToTensor()
        self.mode = mode
        self.hflip = hflip
        self.cropping_factor = cropping_factor
        if self.cropping_factor == 1:
            self.output_reso_h = int(params.reso_h/0.8)
            self.output_reso_w = int(params.reso_w/0.8)
        else:
            self.output_reso_h = int(params.reso_h)
            self.output_reso_w = int(params.reso_w)  


    def __len__(self):
        return len(self.video_paths)
    
    
    def __getitem__(self, index):
        clip, act_label, priv_label, vid_path, frame_list = self.process_data(index)

        if clip is None or priv_label is None or act_label is None:
            print(f"[SKIP] {vid_path}")
            return None  # Skip invalid samples

        return clip, act_label, priv_label, vid_path, frame_list
    
    def process_data(self, idx):
        vid_path = self.video_paths[idx]
        priv_label = self.priv_labels[idx]
        act_class_name = os.path.basename(os.path.dirname(vid_path))
        act_label = self.action_classes.get(act_class_name, None)
        if act_label is None:
            raise ValueError(f"Invalid class name '{act_class_name}' not found in class mapping.")
        act_label -= 1
        
        # clip building:
        clip, frame_list = self.build_clip(vid_path)
        
        return clip, act_label, priv_label, vid_path, idx
    
    
    def build_clip(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            N = len(vr)                          
            T = self.params.all_frames           
            skip = int(self.params.fix_skip)    
            mode = self.mode
            total_modes = self.total_num_modes

            max_required = (T - 1) * skip

            if N <= T:
                # very short video → interpolate
                frames_full = np.linspace(0, N - 1, T).astype(int)

            elif N <= max_required:
                # cannot satisfy original skip → reduce skip
                skip = max(1, N // T)
                frames_full = np.arange(0, skip * T, skip)
                frames_full = np.clip(frames_full, 0, N - 1)

            else:
                # long video → mode-based deterministic start
                available_start = N - max_required   # > 0
                max_start = max(0, available_start - 1)

                if total_modes > 1:
                    # start ∈ [0, max_start]
                    start = int(round(mode * max_start / (total_modes - 1)))
                else:
                    start = 0

                frames_full = start + np.arange(0, skip * T, skip)
                frames_full = np.clip(frames_full, 0, N - 1)

            frames = vr.get_batch(frames_full).permute(0, 3, 1, 2)  # [T, C, H, W]

            self.ori_reso_h, self.ori_reso_w = frames.shape[2], frames.shape[3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            full_clip = []
            for frame in frames:
                full_clip.append(self.augmentation(frame))

            if len(full_clip) < T:
                last = full_clip[-1]
                while len(full_clip) < T:
                    full_clip.append(last.clone())
            full_clip = full_clip[:T]

            return full_clip, frames_full.tolist()

        except Exception as e:
            print(f"[build_clip Exception] {vid_path}: {e}")
            return None, None


    def augmentation(self, image):
        image = self.PIL(image)

        if self.cropping_factor <= 1:
            if self.params.no_ar_distortion:
                image = trans.functional.center_crop(image,(int(self.min_size*self.cropping_factor),int(self.min_size*self.cropping_factor)))
            else:
                image = trans.functional.center_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor)))
                
            if self.threecrop:
                image1 = trans.functional.five_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor))) #torchvision doc says this is non deteministic function, may not always return 5 crops, since I am using bigger overlapping crops, should be fine to just take 2 of the corner crops, let's see how it works. 
                image1_1 = image1[0]
                image1_2 = image1[-2]

        image = trans.functional.resize(image, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.threecrop:
            image1_1 = trans.functional.resize(image1_1, (self.output_reso_h, self.output_reso_w), antialias=True)
            image1_2 = trans.functional.resize(image1_2, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.hflip !=0:
            image = trans.functional.hflip(image)
        if self.threecrop:
            return trans.functional.to_tensor(image), trans.functional.to_tensor(image1_1), trans.functional.to_tensor(image1_2)

        return trans.functional.to_tensor(image)
    
  
  
######------------collate functions -------------######
    
def collate_fn_train(batch):
    clips, act_labels, priv_labels, vid_paths, frame_lists = [], [], [], [], []

    for item in batch:
        if item is None:
            continue

        clip_list, a_lbl, p_lbl, v_path, f_list = item
        if clip_list is None or a_lbl is None or p_lbl is None:
            continue

        if isinstance(clip_list, list):
            clip = torch.stack(clip_list, dim=0).permute(1, 0, 2, 3)
        else:
            clip = clip_list

        clips.append(clip)
        act_labels.append(a_lbl)
        priv_labels.append(p_lbl)
        vid_paths.append(v_path)
        frame_lists.append(f_list)

    if len(clips) == 0:
        return None

    clips = torch.stack(clips, dim=0)
    act_labels = torch.tensor(act_labels)
    priv_labels = torch.tensor(priv_labels)

    return clips, act_labels, priv_labels, vid_paths, frame_lists


def collate_fn_val(batch):
    clips, act_labels, priv_labels, vid_paths, frame_lists = [], [], [], [], []

    for item in batch:
        if item is None:
            continue

        clip_list, a_lbl, p_lbl, v_path, f_list = item
        if clip_list is None or a_lbl is None or p_lbl is None:
            continue

        if isinstance(clip_list, list):
            clip = torch.stack(clip_list, dim=0).permute(1, 0, 2, 3)
        else:
            clip = clip_list

        clips.append(clip)
        act_labels.append(a_lbl)
        priv_labels.append(p_lbl)
        vid_paths.append(v_path)
        frame_lists.append(f_list)

    if len(clips) == 0:
        return None

    clips = torch.stack(clips, dim=0)
    act_labels = torch.tensor(act_labels)
    priv_labels = torch.tensor(priv_labels)

    return clips, act_labels, priv_labels, vid_paths, frame_lists





if __name__ == '__main__':

    train_dataset = vpucf_train_dataloader(params=params,shuffle=True,data_percentage=1.0)
    train_dataloader = DataLoader(train_dataset,batch_size=params.batch_size,shuffle=True,collate_fn=collate_fn_train,num_workers=0)

    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Steps per epoch: {len(train_dataset) // params.batch_size}')

    for i, (clip, act_label, priv_label, vid_path, frame_list) in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            print(f'Full_clip shape is {clip.shape}')
            # print(f'Label is {label}')
            # print(f'Frame list is {frame_list}')
            continue

