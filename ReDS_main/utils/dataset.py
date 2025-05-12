import os
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

torch.set_default_tensor_type(torch.FloatTensor)
class recording_data(Dataset):
    def __init__(self, part='train', datatype='aid'):
        self.part = part
        if datatype == 'fma':
            self.path_to_audio = os.path.join('/your_path/fma_recording_audio/wav/', self.part +'/')
            protocol = os.path.join('/your_path/fma_recording_audio/protocols/', 'recording.'+ self.part + '.txt')
        else:
            self.path_to_audio = os.path.join('/your_path/aid_recording_audio/wav/', self.part +'/')
            protocol = os.path.join('/your_path/aid_recording_audio/protocols/', 'recording.'+ self.part + '.txt')

        self.device_tag = {"mi": 0, "ipad": 1}
        self.distance_tag = {"05": 5, "10": 10, "15": 15}
        self.label = {"recording": 1, "normal": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if datatype == 'fma':
                selected_info = [info for info in audio_info if info[1] == 'ipad' and info[2] == '05' ]
            else:
                if self.part =='test':
                    selected_info = [info for info in audio_info if ('S0001' in info[0] or 'S0002' in info[0]) and info[1] == 'ipad' and info[2] == '05' ]
                else:
                    selected_info = [info for info in audio_info if info[1] == 'ipad' and info[2] == '05' ]
            self.all_info = selected_info
            print(len(self.all_info))
    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, device_tag, distance_tag, label = self.all_info[idx]

        filename_normal = filename[3:]
        wav_normal, sr_normal = sf.read(self.path_to_audio+filename_normal+'.wav')
        wav_recording, sr_recording = sf.read(self.path_to_audio+filename+'.wav')

        wav_normal = torch.from_numpy(wav_normal)[:int(sr_normal*10)]
        wav_recording = torch.from_numpy(wav_recording)[:int(sr_recording*10)]
        
        return {'wav_normal': wav_normal, 'sr_normal': sr_normal,
                'wav_recording': wav_recording, 'sr_recording': sr_recording,
                'recording_device': self.device_tag[device_tag],
                'recording_distance': self.distance_tag[distance_tag],
                'filename':filename}

    def collate_fn(self, samples):
        return default_collate(samples)
        
if __name__ == "__main__":
    validation_set = recording_data('dev')