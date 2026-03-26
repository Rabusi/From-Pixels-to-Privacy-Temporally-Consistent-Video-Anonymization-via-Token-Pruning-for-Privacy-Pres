import os.path 
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Paths for UCF101 dataset.
ucf101_path = './UCF-101/'
ucf101_class_mapping = os.path.join(ucf101_path, 'ucfTrainTestlist', 'action_classes.json')
vp_ucf101_private_label = os.path.join(ucf101_path, 'ucfTrainTestlist', 'VPUCF_annotations', 'vp_ucf101_privacy_attribute_label.csv')

# Paths for HMDB51 dataset.
hmdb51_path = './HMDB51'
hmdb51_class_mapping = os.path.join(hmdb51_path, 'hmdbTrainTestlist', 'action_51_classes.json')
vp_hmdb51_private_label = os.path.join(hmdb51_path, 'hmdbTrainTestlist', 'VPHMDB_annotations', 'vp_hmdb51_privacy_attribute_label.csv')

# General paths.
pretrained_vit_path = './sparsification/evit_timm_pretrained_2_cls.pth'
pretrained_vit_path_1_cls = './Minimization_Noise_1/stprivacy_initialization/evit_timm_pretrained.pth'
saved_models_dir = os.path.join('./', 'saved_models')
logs = os.path.join(saved_models_dir, 'logs')
training_logs = 'logs'
vis_dir_video = os.path.join('./visulization', 'pruned_videos')
vis_dir_cm = os.path.join('./visulization', 'confusion_matrices')
vis_dir = os.path.join('./visulization')


