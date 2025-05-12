import os
import torch

SAVE_MODEL_PATH = 'results'
pre_model_path_ = SAVE_MODEL_PATH + '/model/last_model. dat'
best_model_path_ = SAVE_MODEL_PATH + '/model/best_model.dat'
pre_model_name = "_pre.dat"


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_count=torch.cuda.device_count()
device_ids = list(range(cuda_count))

datatype = 'aid'  # aid or fma
batch_size = 16
epochs = 100
lr = 1e-4  # learning rate
beta_1 = 0.9  # beta_1 for Adam
beta_2 = 0.999  # beta_2 for Adam
eps = 1e-8  # epsilon for Adam
seed = 2023
warmup = True
num_workers = 0

frame_shift = 0.01  # seconds
frame_length = 0.02  # seconds
n_mels = 80  # Number of Mel banks to generate
