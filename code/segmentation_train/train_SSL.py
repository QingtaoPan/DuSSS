# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset_val_SSL import RandomGenerator, ValGenerator, ImageToImage2D, ImageToImage2D_val
from utils.UNet import UNet
from TextMask_generation import TextMask
from torch.utils.data import DataLoader
import logging
from train_SSL_one_epoch import train_one_epoch, print_summary
import Config_SSL as config
from torchvision import transforms
from utils_train import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile


def logger_config(log_path):  # 'MoNuSeg/LViT/Test_session_time/Test_session_time.log'
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def choose_model(model_type, model_):
    if model_type == 'student':
        return model_
    elif model_type == 'teacher':
        for param in model_.parameters():
            param.detach_()
        return model_


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):  # 2,
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])  # 串联图像的多个操作
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
        val_text = read_text(config.test_dataset + 'Test_text.xlsx')  # 验证文本数据 key: values
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_tf,
                                       image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
        val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_tf, image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
    elif config.task_name == 'Covid19_SSL':
        # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        all_text = read_text('/root/data1/lvit_semi_novel/datasets/Covid19/Train_all/Train_text_all.xlsx')  # 训练文本数据 key: value
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_tf, image_size=config.img_size)
    elif config.task_name == 'Bone':
        # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')  # 验证文本数据 key: values
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,  # 2 config.batch_size
                              shuffle=True,  # 每一步打乱顺序
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size_val,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate  # 1e-3
    logger.info(model_type)

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # 4
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # 4
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # 4
        model_1 = UNet()  # 3, 1
        model_2 = UNet()
        model_novel = TextMask()
    else:
        raise TypeError('Please enter a valid name for the model type')
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model_1, inputs=(input, ))  # 计算模型复杂度
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    ######################################################################################################################
    device = torch.device(config.device)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_novel = model_novel.to(device)
    # model_1 = nn.DataParallel(model_1, device_ids=[0, 1])  # 并行运算
    # model_2 = nn.DataParallel(model_2, device_ids=[0, 1])
    # model_novel = nn.DataParallel(model_novel, device_ids=[0, 1])

    #------------------------- 模型参数更新 -------------------------------------- #
    model_student = choose_model(model_type='student', model_=model_1)
    model_teacher = choose_model(model_type='teacher', model_=model_2)

    #------------------------- 模型参数更新 -------------------------------------- #

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=lr)  # Choose optimize
    optimizer_novel = torch.optim.Adam(filter(lambda p: p.requires_grad, model_novel.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler_novel = torch.cuda.amp.GradScaler(enabled=True)

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    global_step = 0

    for epoch in range(config.epochs):  # loop over the dataset multiple times


        global_step = global_step + epoch*len(train_loader)
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model_student.train(True)
        model_teacher.train(True)
        model_novel.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model_student, model_teacher, model_novel, all_text, criterion, optimizer, optimizer_novel, writer, epoch, None, model_type, logger, scaler, scaler_novel, global_step)  # sup

        # evaluate on validation set------------------------------------------------------------------------------------
        logger.info('Validation')
        with torch.no_grad():
            model_student.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model_student, model_teacher, model_novel, all_text, criterion, optimizer, optimizer_novel, writer, epoch, lr_scheduler, model_type, logger, scaler, scaler_novel, global_step)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            logger.info(
                '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
            max_dice = val_dice
            best_epoch = epoch + 1
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model_student.state_dict(),# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model_student


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model_student = main_loop(model_type=config.model_name, tensorboard=True)
