import torch
import os

data_dir = "./dataset" # FMA dataset path
fma_flag = True  # if FMA dataset

payload_length = 100

TRAIN_DATA = 19753
VAL_DATA = 3608
epochs = 100
BATCH = 16
save_circle = 10
WEIGHT_E = 100
WEIGHT_D = 1
data_depth = 2

CRC_LENGTH = 16
CRC_MODULE = 'crc-16'
BCH_POLYNOMIAL = 137 #285
BCH_BITS = 1

SAMPLE_RATE = 48000
NUMBER_SAMPLE= 480000
frame_shift = int(0.01* SAMPLE_RATE)  # seconds
frame_length = int(0.02* SAMPLE_RATE)  # seconds

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0 
cuda_count=torch.cuda.device_count()





