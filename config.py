import datetime

import torch

N_EPOCHS = 50
BATCH_SIZE = 100
DATASET_BASE_PATH = './_dataset'
RAW_DATASET_BASE_PATH = './raw_dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = f'./saves/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
