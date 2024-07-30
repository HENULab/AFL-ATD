import numpy as np
import os
import torch
from config import *
from torchvision import datasets, transforms
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset


def _get_all_mean_std():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    def get_all_data(base_path, data_transforms, batch_size=BATCH_SIZE):
        data_dir = base_path
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'test']}
        all_data = ConcatDataset([image_datasets['train'], image_datasets['test']])
        dataloader = DataLoader(all_data, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)
        return dataloader

    def get_mean_std(loader):
        # Var[x] = E[X**2]-E[X]**2
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1
        # print(num_batches)
        # print(channels_sum)
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean, std

    all_data_dataloader = get_all_data(base_path=RAW_DATASET_BASE_PATH,
                                       data_transforms=data_transforms,
                                       batch_size=BATCH_SIZE)
    all_mean, all_std = get_mean_std(all_data_dataloader)
    logger.info(f"all_mean: {all_mean}, all_std: {all_std}")
    return all_mean, all_std
