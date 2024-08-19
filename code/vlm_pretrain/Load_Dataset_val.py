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


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


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
        image = sample['image']
        image = image.astype(np.uint8)
        image = F.to_pil_image(image)
        x, y = image.size
        if random.random() > 0.5:
            image = random_rot_flip_img(image)
        elif random.random() > 0.5:
            image = random_rotate_img(image)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        image = F.to_tensor(image)
        sample = {'image': image}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
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


class ImageToImage2D_val(Dataset):

    def __init__(self, dataset_path: str, task_name: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.images_list = os.listdir(self.input_path)  # 每个图片名称
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
        image_filename = self.images_list[idx]  # Covid19
        text_index_filename = 'mask_' + image_filename  # Covid19
        image_path = os.path.join(self.input_path, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # correct dimensions if needed
        image = correct_dims(image)  # image[224, 224, 3], mask[224, 224, 1]
        # image_au = image_aug_fun(224)(image)

        sample = {'image': image}  # image[224, 224, 3], mask[224, 224, 1], text[10, 768]

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, text_index_filename  # {'image': image, 'label': mask, 'text': text}, image_filename
