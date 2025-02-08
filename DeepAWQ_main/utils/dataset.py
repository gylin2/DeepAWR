import os
import torchaudio
from utils.hparameter import *
from torch.utils.data import Dataset

def wav_loader(path):
    with open(path, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        sig = sig[0][:NUMBER_SAMPLE]
        return sig, sr


class my_dataset(Dataset):
    def __init__(self, root):
        self.dataset_path = os.path.expanduser(root)
        self.wavs = self.process_meta()

    def __getitem__(self, index):
        path = self.wavs[index]
        audio, sr = wav_loader(path)
        return audio

    def __len__(self):
        return len(self.wavs)

    def process_meta(self):
        wavs = []
        wavs_name = os.listdir(self.dataset_path)
        for name in wavs_name:
            wavs.append(os.path.join(self.dataset_path,name))
        return wavs
