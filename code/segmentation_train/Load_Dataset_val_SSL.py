# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage


def random_rot_flip(image, label, image_unlabeled):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    image_unlabeled = np.rot90(image_unlabeled, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    image_unlabeled = np.flip(image_unlabeled, axis=axis).copy()
    return image, label, image_unlabeled


def random_rotate(image, label, image_unlabeled):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    image_unlabeled = ndimage.rotate(image_unlabeled, angle, order=0, reshape=False)
    return image, label, image_unlabeled


def random_rot_flip_img(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate_img(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image


class image_aug_fun(object):
    def __init__(self, output_size):
        self.output_size = output_size  # [224, 224]

    def __call__(self, sample):
        image = sample
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = image.astype(np.uint8)
        image = F.to_pil_image(image)
        x, y = image.size
        if random.random() > 0.5:
            image = random_rot_flip_img(image)
        elif random.random() > 0.5:
            image = random_rotate_img(image)

        if x != self.output_size or y != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        image = F.to_tensor(image)
        return image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # [224, 224]

    def __call__(self, sample):
        image, label, image_unlabeled = sample['image'], sample['label'], sample['image_unlabeled']
        image, label, image_unlabeled = image.astype(np.uint8), label.astype(np.uint8), image_unlabeled.astype(np.uint8)
        image, label, image_unlabeled = F.to_pil_image(image), F.to_pil_image(label), F.to_pil_image(image_unlabeled)
        x, y = image.size
        if random.random() > 0.5:
            image, label, image_unlabeled = random_rot_flip(image, label, image_unlabeled)
        elif random.random() > 0.5:
            image, label, image_unlabeled = random_rotate(image, label, image_unlabeled)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image_unlabeled = zoom(image_unlabeled, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        image_unlabeled = F.to_tensor(image_unlabeled)
        sample = {'image': image, 'label': label, 'image_unlabeled': image_unlabeled}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img_labeled')
        self.output_path = os.path.join(dataset_path, 'mask_labeled')
        self.images_list = os.listdir(self.input_path)  # 每个图片名称
        self.mask_list = os.listdir(self.output_path)  # 每个标签图片名称
        self.one_hot_mask = one_hot_mask  # False
        self.task_name = task_name  # 'MoNuSeg'

        ##############################################################
        self.input_path_unlabeled = os.path.join(dataset_path, 'img_unlabeled')
        self.images_list_unlabeled = os.listdir(self.input_path_unlabeled)
        ##############################################################

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        # image_filename = self.images_list[idx]  # MoNuSeg
        # mask_filename = image_filename[: -3] + "png"  # MoNuSeg
        mask_filename = self.mask_list[idx]  # Covid19
        image_filename = mask_filename.replace('mask_', '')  # Covid19
        image_path = os.path.join(self.input_path, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        ##############################################################
        image_filename_unlabeled = self.images_list_unlabeled[idx]
        mask_filename_unlabeled = 'mask_' + image_filename_unlabeled
        image_unlabeled_path = os.path.join(self.input_path_unlabeled, image_filename_unlabeled)
        image_unlabeled = cv2.imread(image_unlabeled_path)
        image_unlabeled = cv2.resize(image_unlabeled, (self.image_size, self.image_size))
        ##############################################################

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask, image_unlabeled = correct_dims(image, mask, image_unlabeled)  # image[224, 224, 3], mask[224, 224, 1]
        # image_au = image_aug_fun(224)(image)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'image_unlabeled': image_unlabeled}  # image[224, 224, 3], mask[224, 224, 1], text[10, 768]

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, mask_filename_unlabeled  # {'image': image, 'label': mask, 'text': text}, image_filename


class ImageToImage2D_val(Dataset):

    def __init__(self, dataset_path: str, task_name: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)  # 每个图片名称
        self.mask_list = os.listdir(self.output_path)  # 每个标签图片名称
        self.one_hot_mask = one_hot_mask  # False
        self.task_name = task_name  # 'MoNuSeg'


        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        # image_filename = self.images_list[idx]  # MoNuSeg
        # mask_filename = image_filename[: -3] + "png"  # MoNuSeg
        mask_filename = self.mask_list[idx]  # Covid19
        image_filename = mask_filename.replace('mask_', '')  # Covid19
        image_path = os.path.join(self.input_path, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)  # image[224, 224, 3], mask[224, 224, 1]
        # image_au = image_aug_fun(224)(image)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask}  # image[224, 224, 3], mask[224, 224, 1], text[10, 768]

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, mask_filename  # {'image': image, 'label': mask, 'text': text}, image_filename