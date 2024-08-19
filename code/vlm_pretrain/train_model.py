import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset_val import RandomGenerator, ValGenerator, ImageToImage2D_val
from nets.text_mask.model_novel import my_vlm
from torch.utils.data import DataLoader
import logging
from train_one_epoch import train_one_epoch, print_summary
import Config as config
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



def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):  # 2,
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])  # 串联图像的多个操作
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')  # 验证文本数据 key: values
        train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
        val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
    elif config.task_name == 'Covid19':
        # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_text_all = read_text(config.train_dataset + 'Train_text_all.xlsx')  # 训练文本数据 key: value
        train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_tf, image_size=config.img_size)
    elif config.task_name == 'Bone':
        # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')  # 验证文本数据 key: values
        train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,  # 2 config.batch_size
                              shuffle=True,  # 每一步打乱顺序
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # 4
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # 4
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # 4
        model_novel = my_vlm()

    else:
        raise TypeError('Please enter a valid name for the model type')
    device = torch.device(config.device)
    model_novel = model_novel.to(device)
    input = torch.randn(1, 3, 224, 224).to(device)
    text = ['vision language model']
    flops, params = profile(model_novel, inputs=(input, input, text, ))  # 计算模型复杂度
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    ######################################################################################################################
    # model_novel = nn.DataParallel(model_novel, device_ids=[0, 1])


    #------------------------- 模型参数更新 -------------------------------------- #
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer_novel = torch.optim.Adam(filter(lambda p: p.requires_grad, model_novel.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler_novel = torch.cuda.amp.GradScaler(enabled=True)

    # if config.cosineLR is True:
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer_novel, T_0=10, T_mult=1, eta_min=1e-4)
    # else:
    #     lr_scheduler = None
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
    loss_reference = 100.0

    for epoch in range(config.epochs):  # loop over the dataset multiple times


        global_step = global_step + epoch*len(train_loader)
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model_novel.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        loss = train_one_epoch(train_text_all, train_loader, model_novel, criterion, optimizer_novel, writer, epoch, None, model_type, logger, scaler, scaler_novel, global_step)  # sup

        logger.info('Validation')
        if loss < loss_reference:
            logger.info(
                '\t Saving best model, spine_loss decrease from: {:.4f} to {:.4f}'.format(loss_reference, loss))
            max_dice = 3.33
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model_novel.state_dict(),# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
                             'val_loss': 3.33,
                             'optimizer': optimizer_novel.state_dict()}, config.model_path)
            torch.save(model_novel, config.model_path + '/' + 'best_model.pth')
            loss_reference = loss
            best_epoch = epoch + 1



        if epoch+1 == config.epochs:
            max_dice = 3.33
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'state_dict': model_novel.state_dict(),# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
                             'val_loss': 3.33,
                             'optimizer': optimizer_novel.state_dict()}, config.model_path)
            torch.save(model_novel, config.model_path + '/' + 'last_model.pth')

        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(3.33, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if epoch == config.epochs or early_stopping_count == 50:
            logger.info('\tstopping!')
            break




    return model_novel


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
    model_novel = main_loop(model_type=config.model_name, tensorboard=True)
