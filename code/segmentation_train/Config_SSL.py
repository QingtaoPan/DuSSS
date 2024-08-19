# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'
# use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 100
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 1
early_stopping_patience = 50

area_w = 0.1
tcl_w = 0.1
tv_w = 0.1

pretrain = False
task_name = 'Covid19_SSL'
# task_name = 'MoNuSeg' 
# task_name = 'spine'
# task_name = 'Bone'
learning_rate = 3e-4  # MoNuSeg: 1e-3, Covid19: 3e-4
batch_size = 16  # For LViT-T, 2 is better than 4
batch_size_val = 16

model_name = 'LViT'
# model_name = 'LViT_pretrain'

train_dataset = '/root/data1/lvit_semi_novel/datasets/Covid19/Train_Folder/Train_Folder_0.5_labeled/fold_1/'
val_dataset = '/root/data1/lvit_semi_novel/datasets/Covid19/Val_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"   # 'MoNuSeg/LViT/Test_session_time/Test_session_time.log'
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
# test_session = "Test_session_05.23_14h19"  # dice=79.98, IoU=66.83
