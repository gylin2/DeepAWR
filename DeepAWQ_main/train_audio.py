# -*- coding: utf-8 -*-
import os
import torch
import os.path
import shutil
import random
import datetime
import argparse
import logging
import numpy as np
import torch.nn as nn
from utils.hparameter import *
from distortion_layer.recorder import Recorder
from torch.optim import Adam,  AdamW
from tqdm import tqdm
from networks.encoder import Waveunet as Encoder
from networks.decoder import unet_decoder as Decoder
from utils.dataset import my_dataset
from torch.nn.functional import mse_loss
import warnings
import julius
import torchaudio
# from torch_audiomentations import ApplyImpulseResponse,Compose
from torch.utils.tensorboard import SummaryWriter

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = True

seed = 2023
setup_seed(seed)


warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.info("device: {}".format(device))
logging.info("emebdding rate: {}".format(payload_length))
device_ids = list(range(cuda_count))


def save_model(encoder, decoder, avg_acc, count_epoch, best_flag=False):
    now = datetime.datetime.now()
    name = "epoch_%s_%s_%s.dat" % (count_epoch,now.strftime("%Y-%m-%d_%H_%M_%S"), avg_acc[:6])
    fname = args.save_model_path + '/' + name
    if cuda_count > 1:
        states = {
            'state_dict_encoder': encoder.module.state_dict(),
            'state_dict_decoder': decoder.module.state_dict(),
        }
    else:
        states = {
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
        }
    torch.save(states, fname)
    pre_model_path_ = args.save_model_path+'/last_model.dat'
    torch.save(states, pre_model_path_)
    if best_flag == True:
        best_model_path_ = args.save_model_path+'/best_model.dat'
        torch.save(states, best_model_path_)


def train_model(encoder, decoder, en_de_optimizer, metrics, train_loader, valid_loader):
    writer = SummaryWriter(os.path.join(args.path_to_results, 'tensorboard'))

    recoder = Recorder().to(device)
    if fma_flag == True:
        best_model_path_ = './distortion_layer/fma_best_model.dat'
    else:
        best_model_path_ = './distortion_layer/best_model.dat'

    if torch.cuda.is_available():
        checkpoint = torch.load(best_model_path_, map_location='cuda')
    else:
        checkpoint = torch.load(
            best_model_path_, map_location=lambda storage, loc: storage)
    recoder.load_state_dict(checkpoint['state_dict_recoder'],strict=False)
    recoder.eval()

    spectgram = torchaudio.transforms.Spectrogram(
        n_fft=frame_length, 
        win_length=frame_length,
        hop_length=frame_shift,power=None, 
    ).to(device)
    
    inversespectgram = torchaudio.transforms.InverseSpectrogram(
        n_fft=frame_length, 
        win_length=frame_length,
        hop_length=frame_shift,
    ).to(device)
    
    
    scaler = torch.cuda.amp.GradScaler()
    
    
    encoder.train()
    decoder.train()
    encoder.to(device)
    decoder.to(device)

    start_epoch = 0
    logging.info("cuda count is {} \ncuda count: {}".format(cuda_count,os.environ['CUDA_VISIBLE_DEVICES']))
    if cuda_count > 1:
        encoder = nn.DataParallel(encoder,device_ids=device_ids)
        decoder = nn.DataParallel(decoder,device_ids=device_ids)
        recoder = nn.DataParallel(recoder,device_ids=device_ids)
        

    if fma_flag == True:
        resample = julius.ResampleFrac(44100, 48000).to(device)
    else:
        resample = julius.ResampleFrac(16000, 48000).to(device)

    with open(os.path.join(args.save_log_path, 'train_loss.log'), 'w') as file:
        file.write("epochs_number"+"\t"+"train_steps"+"\t"+"encoder_mse"+"\t"+
                    "decoder_loss"+"\t"+"decoder_acc"+"\n")
    with open(os.path.join(args.save_log_path, 'dev_loss.log'), 'w') as file:
        file.write("epochs_number"+"\t"+"average_encoder_mse"+"\t"+"average_decoder_loss"+"\t"+
                    "average_acc"+"\t"+"best_decoder_loss"+"\t"+"average_snr"+"\n")

    best_loss = 1e8
    for ep in tqdm(range(start_epoch,epochs)):
        metrics['train.encoder_mse'] = []
        metrics['train.decoder_loss'] = []
        metrics['train.decoder_acc'] = []
        metrics['val.encoder_mse'] = []
        metrics['val.decoder_loss'] = []
        metrics['val.decoder_acc'] = []
        metrics['val.snr'] = []
        metrics['val.audio_mse'] = []
        metrics['val.spect_mse'] = []

        logging.info('Epoch {}/{}'.format(ep+1, epochs))
        encoder.train()
        decoder.train()
        recoder.eval()
        for steps, data in enumerate(tqdm(train_loader)):
            with torch.cuda.amp.autocast():
                data = data.to(device)
                data = resample(data)
                spect = spectgram(data)
                amp_spect = torch.abs(spect)
                phase_spect = torch.angle(spect)
                cover = amp_spect[:, :int(amp_spect.size(1)/3), :]
                high_cover = amp_spect[:, int(amp_spect.size(1)/3):, :]
                N, H, W = cover.size()
                payload = torch.zeros((N, 1, payload_length),
                                    device=device).random_(0, 2)
                payload = (payload-0.5)*2
                generated = encoder.forward(cover, payload)


            # --------------Train the generator (encoder-decoder) ---------------------
            en_de_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                
                generated_all = torch.cat([generated, high_cover], dim=1)
                complex_spectrogram = generated_all * torch.exp(1j * phase_spect)
                y = inversespectgram(complex_spectrogram)
                audio_mse = mse_loss(y, data)
                
                spect_y = spectgram(y)
                amp_spect_y = torch.abs(spect_y)
                spect_mse = mse_loss(amp_spect_y, amp_spect)
                encoder_mse = 0.001*spect_mse 

                attacked_data = recoder(y)

                # -------------- DAR&EnvRIR ---------------------
                # augment = Compose([
                #     ApplyImpulseResponse('./utils/irr',p=1,compensate_for_propagation_delay=True),
                #     ])
                # attacked_data = augment(y.unsqueeze(1), sample_rate=SAMPLE_RATE).squeeze(1)
                
                # attacked_data = attacked_data.unsqueeze(1)
                # bandpass = julius.BandPassFilter(1/48, 4/48).to(device)
                # new_audio = bandpass(attacked_data)
                # SNR = random.randint(40,50)
                # mean = 0.
                # RMS_s = torch.sqrt(torch.mean(new_audio**2, dim=2)) 
                # RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
                # for i in range(new_audio.shape[0]):
                #     noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, new_audio.shape[2]))
                #     if i == 0:
                #         batch_noise = noise
                #     else:
                #         batch_noise = torch.cat((batch_noise, noise), dim=0)
                # batch_noise = batch_noise.unsqueeze(1).to(device)
                # attacked_data = new_audio + batch_noise
                # attacked_data = attacked_data.squeeze(1).clamp(-1,1)
                
                # att
                attacked_data = torch.cat([attacked_data, y], dim=0)
                payload = torch.cat([payload, payload], dim=0)

                attack_spect = spectgram(attacked_data)
                attack_amp_spect = torch.abs(attack_spect)
                attack_amp_spect = attack_amp_spect[:, :int(attack_amp_spect.size(1)/3), :]
                decoded = decoder.forward(attack_amp_spect)
                decoder_loss = mse_loss(decoded, payload)
                decoder_acc = (decoded >= 0).eq(
                    payload >= 0).sum().float() / payload.numel()
                metrics['train.encoder_mse'].append(encoder_mse.item())
                metrics['train.decoder_loss'].append(decoder_loss.item())
                metrics['train.decoder_acc'].append(decoder_acc.item())
                loss_all = WEIGHT_E*encoder_mse + decoder_loss

            scaler.scale(loss_all).backward()
            scaler.step(en_de_optimizer)
            scaler.update()

            with open(os.path.join(args.save_log_path, "train_loss.log"), "a") as log:
                log.write(str(ep) + "\t" + str(steps) + "\t" + str(round(encoder_mse.item(),8)) + "\t"+
                        str(round(decoder_loss.item(),8)) + "\t" + str(round(decoder_acc.item(),8)) + "\n")
        
        writer.add_scalar('train_loss/all', loss_all, ep)
        writer.add_scalar('train_loss/audio_mse', WEIGHT_E*audio_mse, ep)
        writer.add_scalar('train_loss/spect_mse', WEIGHT_E*0.001*spect_mse, ep)
        writer.add_scalar('train_loss/encoder_mse', WEIGHT_E*encoder_mse, ep)
        writer.add_scalar('train_loss/decoder_loss', decoder_loss, ep)
        writer.add_scalar('acc/train_acc', decoder_acc, ep)
        
        encoder.eval()
        decoder.eval()
        recoder.eval()
        with torch.no_grad():
            for _, data in enumerate(tqdm(valid_loader)):
                data = data.to(device)
                data = resample(data)
                spect = spectgram(data)
                amp_spect = torch.abs(spect)
                phase_spect = torch.angle(spect)
                cover = amp_spect[:, :int(amp_spect.size(1)/3), :]
                high_cover = amp_spect[:, int(amp_spect.size(1)/3):, :]

                N, H, W = cover.size()
                payload = torch.zeros((N, 1, payload_length),
                                    device=device).random_(0, 2)
                payload = (payload-0.5)*2
                generated = encoder.forward(cover, payload)
                
                generated_all = torch.cat([generated, high_cover], dim=1)
                complex_spectrogram = generated_all * torch.exp(1j * phase_spect)
                y = inversespectgram(complex_spectrogram)

                attacked_data = recoder(y)

                # -------------- DAR&EnvRIR ---------------------
                # augment = Compose([
                #         ApplyImpulseResponse('./utils/irr',p=1,compensate_for_propagation_delay=True),
                #         ])
                # attacked_data = augment(y.unsqueeze(1), sample_rate=SAMPLE_RATE).squeeze(1)

                # attacked_data = attacked_data.unsqueeze(1)
                # bandpass = julius.BandPassFilter(1/48, 4/48).to(device)
                # new_audio = bandpass(attacked_data)
                # SNR = random.randint(40,50)
                # mean = 0.
                # RMS_s = torch.sqrt(torch.mean(new_audio**2, dim=2)) 
                # RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
                # for i in range(new_audio.shape[0]):
                #     noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, new_audio.shape[2]))
                #     if i == 0:
                #         batch_noise = noise
                #     else:
                #         batch_noise = torch.cat((batch_noise, noise), dim=0)
                # batch_noise = batch_noise.unsqueeze(1).to(device)
                # attacked_data = new_audio + batch_noise
                # attacked_data = attacked_data.squeeze(1).clamp(-1,1)

                attacked_data = torch.cat([attacked_data, y], dim=0)
                payload = torch.cat([payload, payload], dim=0)

                attack_spect = spectgram(attacked_data)
                attack_amp_spect = torch.abs(attack_spect)
                attack_amp_spect = attack_amp_spect[:, :int(attack_amp_spect.size(1)/3), :]
                decoded = decoder.forward(attack_amp_spect)
                audio_mse = mse_loss(y, data)

                spect_y = spectgram(y)
                amp_spect_y = torch.abs(spect_y)
                spect_mse = mse_loss(amp_spect_y, amp_spect)
                encoder_mse = 0.001*spect_mse # without audio_mse
            
                decoder_loss = mse_loss(decoded, payload)

                zero_tensor = torch.zeros(data.shape).to(device)
                cover_mse = mse_loss(data, zero_tensor, reduction='sum')
                encoder_mse2 = mse_loss(data, y.clamp(-1,1), reduction='sum')
                snr = 10 * torch.log10(cover_mse / encoder_mse2)

                decoder_acc = (decoded >= 0).eq(
                    payload >= 0).sum().float() / payload.numel()
                
                metrics['val.audio_mse'].append(audio_mse.item())
                metrics['val.spect_mse'].append(spect_mse.item())

                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.snr'].append(snr.item())

        avg_audio_mse = np.mean(metrics['val.audio_mse'])
        avg_spect_mse = np.mean(metrics['val.spect_mse'])

        avg_mse = np.mean(metrics['val.encoder_mse'])
        avg_loss = np.mean(metrics['val.decoder_loss'])
        avg_acc = np.mean(metrics['val.decoder_acc'])
        avg_snr = np.mean(metrics['val.snr'])

        
        avg_metric = avg_loss + WEIGHT_E*avg_mse

        writer.add_scalar('val_loss/all', avg_metric, ep)
        writer.add_scalar('val_loss/audio_mse', WEIGHT_E*avg_audio_mse, ep)
        writer.add_scalar('val_loss/spect_mse', WEIGHT_E*0.001*avg_spect_mse, ep)
        writer.add_scalar('val_loss/encoder_mse', WEIGHT_E*avg_mse, ep)

        writer.add_scalar('val_loss/decoder_loss', avg_loss, ep)
        writer.add_scalar('acc/val_acc', avg_acc, ep)
        writer.add_scalar('snr/val_snr', avg_snr, ep)

        if avg_metric < best_loss:
            best_flag = True
            best_loss = avg_metric
        else:
            best_flag = False
        
        logging.info('\naverage_encoder_mse: {:.8f} - average_decoder_loss: {:.8f} - average_acc: {:.8f} - average_snr: {:.8f} - best_decoder_loss: {:.8f}'.format\
                    (avg_mse, avg_loss, avg_acc, avg_snr, best_loss))
        with open(os.path.join(args.save_log_path, "dev_loss.log"), "a") as log:
            log.write(str(ep) + "\t" + str(round(avg_mse,8)) + "\t" + str(round(avg_loss,8)) + "\t"+
                        str(round(avg_acc,8)) + "\t" + str(round(best_loss,8)) + "\t" + str(round(avg_snr,8)) + "\n")
        
        if best_flag == True:
            save_model(encoder, decoder, str(avg_acc), str(ep+1), best_flag=best_flag)
        if ep >= 80 or ep%save_circle == 0:
            save_model(encoder, decoder, str(avg_acc), str(ep+1), best_flag=best_flag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Watermark Model')
    parser.add_argument('-lr_en_de', '--lr_en_de', type=float,
                        default=1e-4, help='en_de_optimizer learning rate')
    parser.add_argument("-r", "--path_to_results", type=str, help="results path",
                        default='./results/output')
    args = parser.parse_args()
    logging.info("*"*10 + "Train Audio DWT" + "*"*10)
    torch.multiprocessing.set_start_method('spawn')

    args.save_model_path = args.path_to_results+'/model'
    args.save_log_path = args.path_to_results+'/log'

    if os.path.exists(args.path_to_results):
        shutil.rmtree(args.path_to_results)
    if os.path.exists(args.save_model_path):
        shutil.rmtree(args.save_model_path)
    if os.path.exists(args.save_log_path):
        shutil.rmtree(args.save_log_path)

    for func in [
        lambda: os.mkdir(args.path_to_results),
        lambda: os.mkdir(args.save_model_path),
        lambda: os.mkdir(args.save_log_path)]:
        try:
            func()
        except Exception as error:
            print(error)
            continue

    METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.audio_mse',
        'val.spect_mse',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.attacker_mse',
    ]
    
    data_dir = data_dir
    train_set = my_dataset(os.path.join(data_dir, "train/"))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers)
    valid_set = my_dataset(os.path.join(data_dir, "val/"))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH, shuffle=False, num_workers=num_workers)
    
    encoder = Encoder(data_depth)
    decoder = Decoder()
    
    en_de_optimizer = AdamW([
	{'params': decoder.parameters(), 'lr': args.lr_en_de}, 
	{'params': encoder.parameters(), 'lr': args.lr_en_de}
	])

    metrics = {field: list() for field in METRIC_FIELDS}
    train_model(encoder, decoder, en_de_optimizer, metrics, train_loader, valid_loader)
