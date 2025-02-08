import torch
import random
import torch.nn as nn
import random
import julius
from utils.hparameter import *
from audiomentations import Compose, Mp3Compression
import kornia
import numpy as np


class attack_opeartion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.impulse_layer = None
        self.bandpass = julius.BandPassFilter(1/48, 4/48).to(device)
        self.lowpass = julius.LowPassFilter(8/48).to(device)

        self.resample1 = julius.ResampleFrac(SAMPLE_RATE, 16000).to(device)
        self.resample2 = julius.ResampleFrac(16000, SAMPLE_RATE).to(device)

        self.drop_index = torch.ones(NUMBER_SAMPLE, device=self.device)
        i = 0
        while i < NUMBER_SAMPLE:
            self.drop_index[i] = 0.
            i += 100

        self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=64, max_bitrate=64)])

    def white_noise(self, y): # SNR = 10log(ps/pn)
        choice = [20]
        SNR = random.choice(choice)
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit

    def band_pass(self, y):
        y = self.lowpass(y)
        # y = self.band_highpass(y)
        # y = self.band_lowpass(y)
        # y = self.bandpass(y)
        
        return y

    def resample(self, y):
        K = 0.9         
        y = self.resample1(y)
        y = self.resample2(y)
        y = y[:,:,:NUMBER_SAMPLE]
        return y

    def mp3(self, y):
        f = []
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.augment(i,sample_rate=SAMPLE_RATE)[:,:NUMBER_SAMPLE]))
        f = torch.cat(f,dim=0).unsqueeze(1).to(self.device)
        y = y + (f - y)
        return y

    def crop_out(self, y):
        return y*self.drop_index

    def change_top(self, y):
        y = y*0.9
        return y

    def recount(self, y):
        y2 = torch.tensor(np.array(y.cpu().squeeze(0).data.numpy()*(2**7)).astype(np.int8)) / (2**7)
        y2 = y2.to(self.device)
        y = y + (y2 - y)
        return y

    def medfilt(self, y):
        y = kornia.filters.median_blur(y.unsqueeze(1), (1, 3)).squeeze(1)
        return y
    
    def record(self, y):
        # https://github.com/adefossez/julius
        # Impulse Response
        if not IMPULSE_ABLATION:
            y = self.impulse_layer.impulse(y)
        if not BANDPASS_ABLATION:
            y = self.bandpass(y)
        if not NOISE_ABLATION:
            choice = [40, 50]
            # SNR = random.choice(choice)
            SNR = random.randint(40,50)
            mean = 0.
            RMS_s = torch.sqrt(torch.mean(y**2, dim=2)) 
            RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
            for i in range(y.shape[0]):
                noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
                if i == 0:
                    batch_noise = noise
                else:
                    batch_noise = torch.cat((batch_noise, noise), dim=0)
            batch_noise = batch_noise.unsqueeze(1).to(self.device)
            y = y + batch_noise
        return y


    def record2(self, y, global_step):
        ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])
        if not IMPULSE_ABLATION:
            fre = torch.rand(1)[0] * ramp_fn(10000)
            y = self.impulse_layer.impulse(y,fre)

        if not NOISE_ABLATION:
            mean = 0.
            for i in range(y.shape[0]):
                RMS_n = torch.rand(1)[0] * ramp_fn(1000) * 0.02
                # noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
                noise = torch.normal(mean, float(RMS_n), size=(1, y.shape[2]))
                if i == 0:
                    batch_noise = noise
                else:
                    batch_noise = torch.cat((batch_noise, noise), dim=0)
            batch_noise = batch_noise.unsqueeze(1).to(self.device)
            y = y + batch_noise

        if not BANDPASS_ABLATION:
            # high_fre = torch.rand(1)[0] * ramp_fn(10000) * 8/44.1
            fre = torch.rand(1)[0] * ramp_fn(1000)
            if fre > 0.5:
                high_fre = (10 - torch.rand(1)[0] * ramp_fn(10000) * 2)
                band_lowpass = julius.LowPassFilter(high_fre/44.1).to(device)
                y = band_lowpass(y)
            else:
                pass
        return y

    
    def one_white_noise(self, y): # SNR = 10log(ps/pn)
        # choice = [20, 50]
        # SNR = random.choice(choice)
        SNR = random.randint(4,12)*5
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))  # RMS value of signal
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/10)))  # RMS values of noise
        # Therefore mean=0, to round you can use RMS as STD
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit
    
    def two_band_pass(self, y):
        high = random.randint(4,8)
        self.bandpass = julius.LowPassFilter(high/44.1).to(device)
        y = self.bandpass(y)
        return y

    def record3(self, y):
        if not IMPULSE_ABLATION:
            y = self.impulse_layer.impulse(y)
        if not BANDPASS_ABLATION:
            y = self.two_band_pass(y)

        # if not NOISE_ABLATION:
        #     y = self.one_white_noise(y)
        #sample shift
        # shift_len_left = random.randint(0,SAMPLE_RATE*2)
        # shift_len_right = random.randint(0,SAMPLE_RATE*2)
        # shift_nosie_left = torch.normal(mean, float(RMS_n[0][0]), size=(y.shape[0], y.shape(1), shift_len_left))
        # shift_nosie_left = torch.normal(mean, float(RMS_n[0][0]), size=(y.shape[0], y.shape(1), shift_len_right))
        # y = torch.cat([shift_len_left, y, shift_len_right], dim=2)
        return y



    def attack_func(self, y, choice=None):
        '''
        y:[batch, 1, audio_length]
        out:[batch, 1, audio_length]
        '''
        if choice == None:
            return y
        elif choice == 1:
            return self.white_noise(y)
        elif choice == 2:
            return self.band_pass(y)
        elif choice == 3:
            return self.resample(y)
        elif choice == 4:
            return y
        elif choice == 5:
            return self.mp3(y)
        elif choice == 6:
            return self.crop_out(y)
        elif choice == 7:
            return self.change_top(y)
        elif choice == 8:
            return self.recount(y)
        elif choice == 9:
            return self.medfilt(y)
        elif choice == 10:
            return self.record(y)
        elif choice==13: # all attack
            ch = [1,3,4,5,6,7,8,9]
            ch2 = random.choice(ch)
            y = self.attack(y,choice=ch2)
            return y
        elif choice == 20:
            return self.record2(y)
        elif choice == 30:
            return self.record3(y)
        else:
            return y


    def attack(self, y, choice=None):
        y = y.clamp(-1,1)
        if choice==10:
            choice = np.random.choice([0,10])
        out = self.attack_func(y, choice=choice)
        return out.clamp(-1,1)
    