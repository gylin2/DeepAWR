
import torch
import os
import torchaudio
import numpy as np
import soundfile
import warnings
from networks.encoder import Waveunet as RedundantEncoder
from networks.decoder import unet_decoder as RedundantDecoder
from torch.nn.functional import mse_loss
from utils.hparameter import *
warnings.filterwarnings("ignore")
from utils.crc_bch import new_add_bch
from distortion_layer.recorder import Recorder
import julius
from pypesq import pesq
import logging

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

def wm_to_tensor(wm_str):
    data = bytearray(wm_str, 'utf-8')
    data = ''.join(format(x, '08b') for x in data)
    data = torch.Tensor([int(i) for i in data])
    data = new_add_bch(data)
    data = data.view(1,1,-1)[:,:,:payload_length]
    data = (data-0.5)*2
    return data

def do_encode(encoder, sig, sr, watermark, out_path, device, factor):
    with torch.no_grad():
        payload = wm_to_tensor(watermark).to(device)

        spect = spectgram(sig)
        amp_spect = torch.abs(spect).unsqueeze(0)
        phase_spect = torch.angle(spect)
        cover = amp_spect[:, :int(amp_spect.size(1)/3), :]
        high_cover = amp_spect[:, int(amp_spect.size(1)/3):, :]

        generated = encoder.forward(cover, payload, factor)
        generated = torch.cat([generated, high_cover.squeeze(0)], dim=0).squeeze(0)
        complex_spectrogram = generated * torch.exp(1j * phase_spect)
        aud_tensor = inversespectgram(complex_spectrogram)
        
        cover_aud_tensor = sig
        zero_tensor = torch.zeros(cover_aud_tensor.shape).to(device)
        cover_mse = mse_loss(cover_aud_tensor, zero_tensor, reduction='sum')
        encoder_mse = mse_loss(cover_aud_tensor, aud_tensor.clamp(-1,1), reduction='sum')
        snr = 10 * torch.log10(cover_mse / encoder_mse)

        resample_16k = julius.ResampleFrac(SAMPLE_RATE, 16000).to(device)
        aud_tensor_16k = resample_16k(aud_tensor)
        cover_aud_tensor_16k = resample_16k(cover_aud_tensor)
        cur_pesq = pesq(cover_aud_tensor_16k.cpu().squeeze(), aud_tensor_16k.cpu().squeeze(), 16000)

        encoded_audio_for_mel = aud_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
        soundfile.write(out_path, encoded_audio_for_mel, samplerate=sr, subtype='PCM_16')
        print(f"encode finished! encoder_mse:{encoder_mse}, pesq:{cur_pesq}, And snr:{snr}")
    return encoder_mse, snr, generated, payload, aud_tensor, cur_pesq


def do_attack(generated_audio, recoder):
    attacked_data = recoder(generated_audio.unsqueeze(0))
    attacked_data = attacked_data.unsqueeze(1)
    return attacked_data


def do_extract(decoder, attacked_data, payload):
    with torch.no_grad():
        decoded = decoder.forward(attacked_data.squeeze(1))
        decoder_loss = mse_loss(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(
            payload >= 0.0).sum().float() / payload.numel()  # .numel() calculate the number of element in a tensor
        print("Decoder loss: %.3f"% decoder_loss.item())
        print("Decoder acc: %.3f"% decoder_acc.item())
    return decoder_acc.item()


def test_main(audio_name='',
              audio_path='',
              watermark='.',
              audio_out_path='',
              snr_file = None,
              factor=1,
              encoder=None,
              decoder=None,
              recoder = None):
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("\nselected audio:{}\nembeded audio:{}".format(audio_path,audio_out_path))

    p = audio_path
    # load audio data and transform
    with open(p, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        sig = torch.FloatTensor(sig).to(device)
        sig = sig[0][:int(sr*10)]
        if sr != SAMPLE_RATE:
            resample = julius.ResampleFrac(sr, SAMPLE_RATE).to(device)
            sig = resample(sig)

    # do encode
    out_path = audio_out_path
    encoder_mse, snr, generated, payload, generated_audio, cur_pesq = do_encode(encoder, sig, SAMPLE_RATE, watermark, out_path, device, factor)
    snr_file.write(f"{audio_name}\t{snr}\n")
    attacked_data = do_attack(generated_audio, recoder)
    # do extract
    attack_spect = spectgram(attacked_data)
    attack_amp_spect = torch.abs(attack_spect)[:, :, :int(attack_spect.size(2)/3), :]
    acc = do_extract(decoder, attack_amp_spect, payload)
    return snr, acc, cur_pesq


watermark ='helloworld!!!' # 100bit
# watermark ='helloworld!!!helloworld!!!' # 200bit
# watermark ='helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!' # 500bit
# watermark ='helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!' # 1000bit

def main(root = './your_testing_dataset', out_path = './results/test_result', watermark=watermark, factor=1):
    out = out_path + '/test_result'
    SNR_FILE = os.path.join(out, 'snr.txt')
    AUDIO_OUT_PATH = os.path.join(out, "wm_audio/")
    ATTACK_OUT_PATH = os.path.join(out, "attack_audio/")

    SPECTRUM_PATH = os.path.join(out, "spectrum/")
    files = os.listdir(root)
    files.sort()
    audio_list = []
    for file_ in files:
        if file_[-4:] == ".wav" or file_[-4:] == "flac":
            audio_list.append(file_)
    if not os.path.exists(AUDIO_OUT_PATH): os.makedirs(AUDIO_OUT_PATH)
    if not os.path.exists(ATTACK_OUT_PATH): os.makedirs(ATTACK_OUT_PATH)
    if not os.path.exists(SPECTRUM_PATH): os.makedirs(SPECTRUM_PATH)
    snr_file = open(SNR_FILE,'a+')
    model_path = out_path+'/model'
    model_list = []
    files = os.listdir(model_path)

    for file_ in files:
        if file_[-4:] == ".dat":
            model_list.append(file_)
    PATH = os.path.join(model_path, 'best_model.dat')
    print(PATH)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = RedundantEncoder(data_depth).to(device)
    decoder = RedundantDecoder().to(device)
    recoder = Recorder(device).to(device)
    # load net parameters
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location='cuda')
    else:
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)

    encoder.load_state_dict(checkpoint['state_dict_encoder'],strict=False)
    decoder.load_state_dict(checkpoint['state_dict_decoder'],strict=False)
    if fma_flag == True:
        best_model_path_ = './distortion_layer/fma_best_model.dat'
    else:
        best_model_path_ = './distortion_layer/best_model.dat'
    if torch.cuda.is_available():
        checkpoint2 = torch.load(best_model_path_, map_location='cuda')
    else:
        checkpoint2 = torch.load(
            best_model_path_, map_location=lambda storage, loc: storage)
      
    recoder.load_state_dict(checkpoint2['state_dict_recoder'],strict=False)

    avg_snr = 0
    avg_acc = 0
    number = 0
    pesq_list = []
    for SELECT_AUDIO_NAME in audio_list:
        TEST_AUDIO = os.path.join(root,SELECT_AUDIO_NAME)
        EMBED_AUDIO = os.path.join(AUDIO_OUT_PATH, SELECT_AUDIO_NAME)
        if fma_flag == False:
            if 'S0001' not in TEST_AUDIO and 'S0002' not in TEST_AUDIO:
                continue
        snr, acc, cur_pesq = test_main(
                        audio_name=SELECT_AUDIO_NAME,
                        audio_path=TEST_AUDIO,
                        watermark=watermark,
                        audio_out_path=EMBED_AUDIO,
                        snr_file = snr_file,
                        factor = factor,
                        encoder=encoder,
                        decoder=decoder,
                        recoder = recoder)
        avg_snr += snr
        avg_acc += acc
        pesq_list.append(cur_pesq)
        number += 1
        if number >= 200 and fma_flag == True:
            break
    avg_snr /= number
    avg_acc /= number
    avg_pesq = np.nanmean(pesq_list)
    snr_file.write(f"{avg_snr} {avg_acc} {avg_pesq}\n")
    print('avg_snr: {}, avg_pesq: {}'.format(avg_snr, avg_pesq))
    snr_file.close()

if __name__ == "__main__":
    import fire
    fire.Fire(main)