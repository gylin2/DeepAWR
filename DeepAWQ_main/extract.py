
import torch
import os
import torchaudio
import numpy as np
import warnings
from networks.decoder import unet_decoder as RedundantDecoder
from torch.nn.functional import mse_loss
warnings.filterwarnings("ignore")
from utils.crc_bch import new_add_bch, verify_crc, do_ec
from utils.hparameter import *
import logging
import julius
import kornia
from pydub import AudioSegment
import io


logging.basicConfig(level=logging.INFO, format='%(message)s')
spectgram = torchaudio.transforms.Spectrogram(
        n_fft=frame_length, 
        win_length=frame_length,
        hop_length=frame_shift,power=None, 
    ).to(device)

mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000, n_fft=frame_length, 
        hop_length=frame_shift, win_length=frame_length, 
        n_mels=128, f_max=24000).to(device)

def wm_to_tensor(wm_str):
    data = bytearray(wm_str, 'utf-8')
    data = ''.join(format(x, '08b') for x in data)
    data = torch.Tensor([int(i) for i in data])
    data = new_add_bch(data)
    data = data.view(1,1,-1)[:,:,:payload_length]
    data = (data-0.5)*2
    return data

def white_noise(y, choice=20): # SNR = 10log(ps/pn)
    SNR = choice
    mean = 0.
    RMS_s = torch.sqrt(torch.mean(y**2))
    RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/10)))

    noise = torch.normal(mean, float(RMS_n), size=(y.shape)).to(device)
    signal_edit = y + noise
    return signal_edit

def mp3(raw_sig, bitrate="128k"):
    int_audio = np.int16(raw_sig.cpu() * 32767)
    # Convert the NumPy array to an AudioSegment object
    audio = AudioSegment(
        int_audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=int_audio.dtype.itemsize,
        channels=1
    )
    # Use BytesIO as an in-memory buffer to avoid disk I/O operations
    buffer = io.BytesIO()
    # Export the audio as MP3 format with the specified bitrate
    audio.export(buffer, format="mp3", bitrate=bitrate)
    # Reset the buffer's pointer to the start to read from it
    buffer.seek(0)
    # Load the MP3 audio from the buffer
    loaded_mp3 = AudioSegment.from_mp3(buffer)
    # Convert the loaded audio (AudioSegment) back into a PyTorch tensor
    loaded_mp3 = torch.tensor(loaded_mp3.get_array_of_samples(), dtype=torch.float32).unsqueeze(0).to(device)
    return (loaded_mp3/32768.0)

def low_pass(y):
    lowpass = julius.LowPassFilter(8/48).to(device)
    y = lowpass(y)
    return y

def resample(y, K=44100):
    resample1 = julius.ResampleFrac(SAMPLE_RATE, K).to(device)
    resample2 = julius.ResampleFrac(K, SAMPLE_RATE).to(device)   
    y = resample1(y)
    y = resample2(y)
    return y

def crop_out(y):
    drop_index = torch.ones(NUMBER_SAMPLE, device=device)
    i = 0
    while i < NUMBER_SAMPLE:
        drop_index[i] = 0.
        i += 100
    return y*drop_index

def change_top(y):
    y = y*0.9
    return y

def recount(y):
    y2 = torch.tensor(np.array(y.cpu().data.numpy()*(2**7)).astype(np.int8)) / (2**7)
    y2 = y2.to(device)
    y = y + (y2 - y)
    return y

def medfilt(y):
    y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y = kornia.filters.median_blur(y, (1, 3)).squeeze()
    return y



def search_do_extract(attacked_path, watermak, decoder, len_file):
    p = attacked_path
    with open(p, 'rb') as f:
        raw_sig, sr = torchaudio.load(f.name)
        raw_sig = torch.FloatTensor(raw_sig).to(device)
        if sr != SAMPLE_RATE:
            resampler = julius.ResampleFrac(sr, SAMPLE_RATE).to(device)
            raw_sig = resampler(raw_sig)

    max_start = raw_sig.shape[1] - NUMBER_SAMPLE
    max_len = int(-6000) 
    start_len = int(-12000) 
    payload = wm_to_tensor(watermak).to(device)
    best_acc = 0
    for i_len in range(start_len, max_len):
        if i_len > 0:
            sig = torch.cat([torch.zeros(1,i_len),raw_sig], dim=1)
            sig = sig[0][:NUMBER_SAMPLE]
        else:
            if max_start + i_len < 0:
                sig = torch.cat([raw_sig,torch.zeros(1,-i_len-max_start)], dim=1)
                sig = sig[0][-i_len:NUMBER_SAMPLE-i_len]
            else:
                sig = raw_sig[0][-i_len:NUMBER_SAMPLE-i_len]
        attack_spect = spectgram(sig)
        sig = torch.abs(attack_spect).unsqueeze(0)[:, :int(attack_spect.size(0)/3), :]
        with torch.no_grad():
            decoded = decoder.forward(sig)
            decoder_acc = (decoded >= 0.0).eq(payload >= 0.0).sum().float() / payload.numel() 
            
            if best_acc < decoder_acc:
                best_decoded = decoded
                best_acc = decoder_acc
            if best_acc == 1:
                best_decoded = decoded
                break
            
    print('acc: {}'.format(best_acc))
    decoded = (best_decoded >= 0.0).float()
    decoded_ec = do_ec(decoded.view(-1))
    verify = verify_crc(decoded_ec)
    decoder_acc = best_acc
    return decoder_acc


def extract(attacked_path, watermak, decoder, len_file, attack=0):
    p = attacked_path
    # load audio data and transform
    with open(p, 'rb') as f:
        raw_sig, sr = torchaudio.load(f.name)
        raw_sig = torch.FloatTensor(raw_sig).to(device)
        cover_aud_tensor = raw_sig.squeeze()
        if sr != SAMPLE_RATE:
            print("resample")
            resampler = julius.ResampleFrac(sr, SAMPLE_RATE).to(device)
            raw_sig = resampler(raw_sig)
    raw_sig = raw_sig[0][:NUMBER_SAMPLE]
    if attack == 11:
        raw_sig = white_noise(raw_sig,20)
    elif attack == 12:
        raw_sig = white_noise(raw_sig,30)
    elif attack == 13:
        raw_sig = white_noise(raw_sig,40)
    elif attack == 14:
        raw_sig = white_noise(raw_sig,50)
    elif attack == 21:
        raw_sig = mp3(raw_sig,bitrate='64k')
    elif attack == 22:
        raw_sig = mp3(raw_sig,bitrate='128k')
    elif attack == 3:
        raw_sig = low_pass(raw_sig)
    elif attack == 41:
        raw_sig = resample(raw_sig, K=16000)
    elif attack == 42:
        raw_sig = resample(raw_sig, K=44100)
    elif attack == 5:
        raw_sig = crop_out(raw_sig)
    elif attack == 6:
        raw_sig = change_top(raw_sig)
    elif attack == 7:
        raw_sig = recount(raw_sig)
    elif attack == 8:
        raw_sig = medfilt(raw_sig)
    sig = raw_sig.squeeze()[:NUMBER_SAMPLE]
    zero_tensor = torch.zeros(cover_aud_tensor.shape).to(device)
    cover_mse = mse_loss(cover_aud_tensor, zero_tensor, reduction='sum')
    encoder_mse = mse_loss(cover_aud_tensor, sig.clamp(-1,1), reduction='sum')
    snr = 10 * torch.log10(cover_mse / encoder_mse)

    payload = wm_to_tensor(watermak).to(device)
    attack_spect = spectgram(sig)
    sig = torch.abs(attack_spect).unsqueeze(0)[:, :int(attack_spect.size(0)/3), :]
    with torch.no_grad():
        decoded = decoder.forward(sig)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.0).sum().float() / payload.numel()
        decoded = (decoded >= 0.0).float()
        # decoded_ec = do_ec(decoded.view(-1))
        # verify = verify_crc(decoded_ec)
    
        decoder_loss = mse_loss(decoded, payload)
        logging.info("Decoder loss: %.3f, Decoder acc: %.3f, Decoder snr: %.3f"% (
            decoder_loss.item(),decoder_acc.item(), snr.item()))
        
        len_file.write("{}\t{}\{}\t{}\n".format(os.path.basename(p),decoder_acc,decoder_loss,0))
    return decoder_acc, snr

watermark ='helloworld!!!' # 100bit
# watermark ='helloworld!!!helloworld!!!' # 200bit
# watermark ='helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!' # 500bit
# watermark ='helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!helloworld!!!' # 1000bit
def main(attack=0, root="./results/test_result/attacked/", dir="./results", watermark = watermark):
    #---------------------------get model--------------------------
    model_path = dir + '/model/'
    acc_dir = dir+'/test_result/'
    model_list = []
    files = os.listdir(model_path)
    files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    for file_ in files:
        if file_[-4:] == ".dat":
            model_list.append(file_)

    SNR_FILE = os.path.join(acc_dir, 'output.txt')
    output_file = open(SNR_FILE,'a+')
    output_file.write(root + '\n')

    PATH = os.path.join(model_path, 'best_model.dat')
    # PATH = os.path.join(model_path, 'last_model.dat')
    logging.info("model:\n{}".format(PATH))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = RedundantDecoder().to(device)
    # load net parameters
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location='cuda')
    else:
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(checkpoint['state_dict_decoder'],strict=False)

    #---------------------------get AUDIO--------------------------
    files = os.listdir(root)
    files.sort()
    audio_list = []
    for file_ in files:
        if file_[-4:] == ".wav" or file_[-4:] == ".mp3" or file_[-4:] == "flac":
            audio_list.append(file_)
    ACC_FILE = os.path.join(acc_dir, "acc.txt")
    acc_file = open(ACC_FILE,'a+')
    #---------------------------get WATERMARK--------------------------
    #---------------------------Extract watermark--------------------------
    if attack == -1:
        attack_list = [0,11,12,13,14,21,22,3,41,42,5,6,7,8]
    else:
        attack_list = [attack]
    all_acc_list = []
    for attack_id in attack_list:
        avg_acc = 0
        avg_snr = 0
        for SELECT_AUDIO_NAME in audio_list:
            attacked_path = os.path.join(root, SELECT_AUDIO_NAME)
            logging.info("\nattacked audio:{}".format(attacked_path))
            if attack_id == -2:
                acc = search_do_extract(attacked_path, watermark, decoder, acc_file)
                snr = acc
            else:
                acc, snr = extract(attacked_path, watermark, decoder, acc_file, attack_id)
            avg_acc += acc
            avg_snr += snr
        avg_acc /= len(audio_list)
        avg_snr /= len(audio_list)
        acc_file.write(f"{avg_acc, avg_snr}\n")
        print('acc: {}, snr: {}'.format(avg_acc, avg_snr))
        all_acc_list.append(avg_acc)
        output_file.write(str(avg_acc.item())+ '\n')

if __name__ == "__main__":
    import fire
    fire.Fire(main)
