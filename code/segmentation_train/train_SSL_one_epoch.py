# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim
import os
import time
import Config_SSL as config
import warnings
import numpy as np
from utils_train import *


warnings.filterwarnings("ignore")

def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger, novel_loss):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    string += 'novel_loss:{:.4f}'.format(novel_loss)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def update_teacher_model(model_student, model_teacher, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for tea_param, stu_param in zip(model_teacher.parameters(), model_student.parameters()):
        tea_param.data.mul_(alpha).add_(1 - alpha, stu_param.data)


def train_one_epoch(loader, model_student, model_teacher, model_novel, all_text, criterion, optimizer, optimizer_novel, writer, epoch, lr_scheduler, model_type, logger, scaler, scaler_novel, global_step):

    device = torch.device(config.device)
    logging_mode = 'Train' if model_student.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    loss_novel_sum = 0.0
    dices = []
    if logging_mode == 'Train':
        for i, (sampled_batch, names) in enumerate(loader, 1):

            try:
                loss_name = criterion._get_name()
            except AttributeError:
                loss_name = criterion.__name__

            # Take variable and put them to GPU
            images, mask_lab, images_un = sampled_batch['image'], sampled_batch['label'], sampled_batch['image_unlabeled']
            images, mask_lab, images_un = images.to(device), mask_lab.to(device), images_un.to(device)   # images[b, 3, 224, 224], masks[b, 224, 224]
            images_all = torch.cat([images, images_un])

            # 获取文本数据--------------------------------------------------------------------
            text_str = names

            with torch.cuda.amp.autocast(enabled=True):
                preds_all = model_student(images_all)  # [b, 1, 224, 224]
                loss_novel, text_mask = model_novel(images_un, text_str)
                # loss_novel = torch.tensor([2.0], requires_grad=True)
                mask_unlab = model_teacher(images_un)
                mask_unlab = mask_unlab.squeeze(1)
                text_mask = text_mask.squeeze(1)
                mask_merge = mask_unlab + text_mask
                mask_merge = nn.Sigmoid()(mask_merge)

            mask_lab_unlab = torch.cat([mask_lab, mask_merge])

            loss_lab_unlab = criterion(preds_all, mask_lab_unlab.half())  # Loss_value = ([b, 1, 224, 224], [b, 1, 224, 224])
            loss_textmask2pseudo = criterion(text_mask.half(), mask_unlab.half())


            loss_ = (loss_lab_unlab.item() + loss_textmask2pseudo.item()) / 2.0
            loss_all = (loss_lab_unlab + loss_textmask2pseudo) / 2.0

            if model_student.training:
                optimizer.zero_grad()
                scaler.scale(loss_all).backward(retain_graph=True)
                # scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                update_teacher_model(model_student, model_teacher, alpha=0.99, global_step=global_step)
                #######################
                optimizer_novel.zero_grad()
                scaler_novel.scale(loss_novel).backward()
                scaler_novel.unscale_(optimizer_novel)
                scaler_novel.step(optimizer_novel)
                scaler_novel.update()
                #--------------------------
                loss_all.detach_()
                preds_all.detach_()
                mask_lab_unlab.detach_()


            train_dice = criterion._show_dice(preds_all, mask_lab_unlab.half())
            train_iou = iou_on_batch(mask_lab_unlab, preds_all)

            batch_time = time.time() - end
            # if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            #     vis_path = config.visualize_path+str(epoch)+'/'
            #     if not os.path.isdir(vis_path):
            #         os.makedirs(vis_path)
            #     save_on_batch(images,masks,preds.float(),names,vis_path)
            dices.append(train_dice)

            time_sum += len(images) * batch_time
            loss_sum += len(images) * loss_
            iou_sum += len(images) * train_iou
            # acc_sum += len(images) * train_acc
            dice_sum += len(images) * train_dice
            loss_novel_sum += len(images)/2 * loss_novel.item()

            if i == len(loader):
                average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
                average_time = time_sum / (config.batch_size*(i-1) + len(images))
                train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
                # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
                train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
                loss_novel__avg = loss_novel_sum / (config.batch_size*(i-1) + len(images)/2)
            else:
                average_loss = loss_sum / (i * config.batch_size)
                average_time = time_sum / (i * config.batch_size)
                train_iou_average = iou_sum / (i * config.batch_size)
                # train_acc_average = acc_sum / (i * config.batch_size)
                train_dice_avg = dice_sum / (i * config.batch_size)
                loss_novel__avg = loss_novel_sum / (i * config.batch_size)

            end = time.time()
            if i % config.print_frequency == 0:
                print_summary(epoch + 1, i, len(loader), loss_, loss_name, batch_time,
                              average_loss, average_time, train_iou, train_iou_average,
                              train_dice, train_dice_avg, 0, 0,  logging_mode,
                              lr=min(g["lr"] for g in optimizer.param_groups),logger=logger, novel_loss=loss_novel__avg)

            if config.tensorboard:
                step = epoch * len(loader) + i
                writer.add_scalar(logging_mode + '_' + loss_name, loss_, step)  # loss.item()改为loss

                # plot metrics in tensorboard
                writer.add_scalar(logging_mode + '_iou', train_iou, step)
                # writer.add_scalar(logging_mode + '_acc', train_acc, step)
                writer.add_scalar(logging_mode + '_dice', train_dice, step)


        if lr_scheduler is not None:
            lr_scheduler.step()

    if logging_mode == 'Val':
        for i, (sampled_batch, names) in enumerate(loader, 1):

            try:
                loss_name = criterion._get_name()
            except AttributeError:
                loss_name = criterion.__name__

            # Take variable and put them to GPU
            images, masks = sampled_batch['image'], sampled_batch['label']

            # 将数据分为: 有标签-无标签
            images, masks = images.to(device), masks.to(device)  # images[b, 3, 224, 224], masks[b, 224, 224], text[b, 10, 768]

            with torch.cuda.amp.autocast(enabled=True):
                preds = model_student(images)  # [b, 1, 224, 224]

            loss_out = criterion(preds, masks.half())  # Loss_value = ([b, 1, 224, 224], [b, 1, 224, 224])
            loss_ = loss_out.item()

            train_dice = criterion._show_dice(preds, masks.half())
            train_iou = iou_on_batch(masks, preds)

            batch_time = time.time() - end
            if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
                vis_path = config.visualize_path+str(epoch)+'/'
                if not os.path.isdir(vis_path):
                    os.makedirs(vis_path)
                save_on_batch(images,masks,preds.float(),names,vis_path)
            dices.append(train_dice)

            time_sum += len(images) * batch_time
            loss_sum += len(images) * loss_
            iou_sum += len(images) * train_iou
            # acc_sum += len(images) * train_acc
            dice_sum += len(images) * train_dice

            if i == len(loader):
                average_loss = loss_sum / (config.batch_size_val*(i-1) + len(images))
                average_time = time_sum / (config.batch_size_val*(i-1) + len(images))
                train_iou_average = iou_sum / (config.batch_size_val*(i-1) + len(images))
                # train_acc_average = acc_sum / (config.batch_size_val*(i-1) + len(images))
                train_dice_avg = dice_sum / (config.batch_size_val*(i-1) + len(images))
            else:
                average_loss = loss_sum / (i * config.batch_size_val)
                average_time = time_sum / (i * config.batch_size_val)
                train_iou_average = iou_sum / (i * config.batch_size_val)
                # train_acc_average = acc_sum / (i * config.batch_size_val)
                train_dice_avg = dice_sum / (i * config.batch_size_val)

            end = time.time()
            if i % config.print_frequency == 0:
                print_summary(epoch + 1, i, len(loader), loss_, loss_name, batch_time,
                              average_loss, average_time, train_iou, train_iou_average,
                              train_dice, train_dice_avg, 0, 0,  logging_mode,
                              lr=min(g["lr"] for g in optimizer.param_groups),logger=logger, novel_loss=1.0)

            if config.tensorboard:
                step = epoch * len(loader) + i
                writer.add_scalar(logging_mode + '_' + loss_name, loss_, step)  # loss.item()改为loss

                # plot metrics in tensorboard
                writer.add_scalar(logging_mode + '_iou', train_iou, step)
                # writer.add_scalar(logging_mode + '_acc', train_acc, step)
                writer.add_scalar(logging_mode + '_dice', train_dice, step)


        if lr_scheduler is not None:
            lr_scheduler.step()

    return average_loss, train_dice_avg





