# -*- coding: utf-8 -*-
import os
import torch
import os.path
import shutil
import logging
import numpy as np
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import julius
from hparameter import *
from recorder import Recorder
from utils.dataset import recording_data
from utils.util import setup_seed
from tqdm import tqdm


def save_model(recoder, avg_acc, prev_loss):
    if cuda_count > 1:
        states = {
            'state_dict_recoder': recoder.module.state_dict(),
        }
    else:
        states = {
            'state_dict_recoder': recoder.state_dict(),
        }
    torch.save(states, pre_model_path_)
    if avg_acc < prev_loss:
        torch.save(states, best_model_path_)


def adjust_learning_rate(optimizer, epoch_num, n_current_steps=0):
    if warmup:
        n_warmup_steps=1000
        lr = np.power(64, -0.5) * np.min([
                np.power(n_current_steps, -0.5),
                np.power(n_warmup_steps, -1.5) * n_current_steps])
    else:
        lr = lr * (0.5 ** (epoch_num // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model():
    setup_seed(seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    recoder = Recorder().to(device)


    if cuda_count > 1:
        logging.info("cuda count is {} \ncuda count: {}".format(cuda_count,os.environ['CUDA_VISIBLE_DEVICES']))
        recoder = nn.DataParallel(recoder, device_ids=device_ids)
    

    wave_optimizer = torch.optim.Adam(recoder.parameters(), lr=lr,
                                      betas=(beta_1, beta_2), eps=eps, 
                                      weight_decay=0, amsgrad=True)
    
    training_set = recording_data('train', datatype)
    validation_set = recording_data('dev', datatype)                          
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 collate_fn=training_set.collate_fn, pin_memory=False)
    valid_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               collate_fn=validation_set.collate_fn, pin_memory=False)
    batch = training_set[29]
    print("Feature shape", batch['wav_normal'].shape)

    mfcc_recording_transform  = torchaudio.transforms.MFCC(sample_rate=batch['sr_recording'],n_mfcc=40, melkwargs={
                                "n_fft":int(batch['sr_recording']*frame_length),
                                "hop_length":int(batch['sr_recording']*frame_shift),
                                "win_length":int(batch['sr_recording']*frame_length),
                                "n_mels":n_mels, "f_max":batch['sr_recording']//2}).to(device)
    
    if datatype == 'aid':                       
        resample = julius.ResampleFrac(16000, 48000).to(device)
    elif datatype == 'fma':
        resample = julius.ResampleFrac(44100, 48000).to(device)
        
    start_epoch = 0
    prev_loss = 1e8
    METRIC_FIELDS = [
        'train.recoder_loss',
        'val.recoder_loss',
    ]
    metrics = {field: list() for field in METRIC_FIELDS}
    for ep in tqdm(range(start_epoch, epochs)):
        metrics['train.recoder_loss'] = []
        metrics['val.recoder_loss'] = []
        metrics['val.rmse'] = []
        logging.info('Epoch {}/{}'.format(ep+1, epochs))
        step = 1
        recoder.train()

        for _, batch in enumerate(tqdm(train_loader)):
            wav_normal = batch['wav_normal'].float().to(device)
            wav_recording = batch['wav_recording'].float().to(device)

            wav_normal = resample(wav_normal)
            wav_predict = recoder(wav_normal)
            
            mfcc_predict= mfcc_recording_transform(wav_predict)
            mfcc_recording = mfcc_recording_transform(wav_recording)
        
            wave_loss = torch.mean(torch.sqrt(torch.mean((mfcc_predict-mfcc_recording)**2,dim=1)))


            wave_optimizer.zero_grad()
            metrics['train.recoder_loss'].append(wave_loss.item())
            wave_loss.backward()
            wave_optimizer.step()
            step += 1

            if warmup:
                adjust_learning_rate(wave_optimizer, ep, step)
            

        
        recoder.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(valid_loader)):
                wav_normal = batch['wav_normal'].float().to(device)
                wav_recording = batch['wav_recording'].float().to(device)      

                wav_normal = resample(wav_normal)
                wav_predict = recoder(wav_normal)
                
                mfcc_predict= mfcc_recording_transform(wav_predict)
                mfcc_recording = mfcc_recording_transform(wav_recording)

                wave_loss = torch.mean(torch.sqrt(torch.mean((mfcc_predict-mfcc_recording)**2,dim=1)))
                # wave_loss = mse_loss(mfcc_predict, mfcc_recording)
                # wave_loss = torch.sqrt(mse_loss(mfcc_predict, mfcc_recording))

                metrics['val.recoder_loss'].append(wave_loss.item())

            with open(os.path.join(SAVE_MODEL_PATH, "dev_loss.log"), "a") as log:
                log.write(str(ep) + "\t" + str(np.nanmean(metrics['val.recoder_loss'])) + "\t" +"\n")
            val_loss = np.nanmean(metrics['val.recoder_loss'])
            logging.info('\n val_decoder_loss: {}'.format(val_loss))

        
        save_model(recoder, val_loss, prev_loss)
        if val_loss < prev_loss:
            prev_loss = val_loss


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    if os.path.exists(os.path.join('.', SAVE_MODEL_PATH)):
        shutil.rmtree(os.path.join('.', SAVE_MODEL_PATH))

    for func in [
        lambda: os.mkdir(os.path.join('.', SAVE_MODEL_PATH)),
        lambda: os.mkdir(os.path.join('.', SAVE_MODEL_PATH, 'model'))]:
        try:
            func()
        except Exception as error:
            print(error)
            continue
    

    train_model()
