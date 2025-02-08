import numpy as np
import soundfile as sf
import os
import librosa
import librosa.display
from tqdm import tqdm

def mse(target, predict):
    # Mean Squared Error (MSE) calculation
    return ((target - predict)**2).mean()

# Generate Mel spectrogram with specified parameters
def feature_melspect(y, sr, n_mels=128, n_fft=2048, hop_length=1024):
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, 
                                               n_fft=n_fft, 
                                               hop_length=hop_length)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect

if __name__ == "__main__":
    
    # Path to re-recorded audio files
    path_to_audio = "re-record_150cm.m4a"

    # FFT window size and hop size in milliseconds
    fft_ms = 20 
    hop_ms = 10

    # Load audio using librosa
    wav, sr = librosa.load(path_to_audio, sr=None)
    print('Read over')

    # Set batch and overlap sizes (in samples)
    batch_duration = 120  # seconds
    overlap_duration = 0.01  # seconds
    batch_samples = int(batch_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    
    mel_spect_batches = []
    i = 0
    # Loop over the audio in batches
    for start in tqdm(range(0, len(wav), int(batch_samples - overlap_samples))):
        end = min(start + batch_samples, len(wav))
        batch_wav = wav[start:end]
        mel_spect_batch = feature_melspect(batch_wav, sr, n_fft=int(fft_ms * sr / 1000), 
                                           hop_length=int(hop_ms * sr / 1000))
        if i == 0:
            mel_spect_batches = mel_spect_batch[:, :-1]
        else:
            if start + batch_samples > len(wav):
                mel_spect_batch = mel_spect_batch[:, 1:]
            else:
                mel_spect_batch = mel_spect_batch[:, 1:-1]
            mel_spect_batches = np.concatenate((mel_spect_batches, mel_spect_batch), axis=-1)
        i += 1
    mel_spect_0 = mel_spect_batches
    print('batch read over')
    
    pos = 0
    name = 0
    name_number = 1
    
    # Path to audio classes
    audio_path = "your_paths"
    class_list = ["fma", "zh"]

    # Process each class
    number = 0
    for class_name in class_list:
        this_audio_path = os.path.join(audio_path, class_name, 'wm_audio')
        path_to_output = os.path.join(audio_path, class_name, 'attack_audio_150cm')
        
        if not os.path.exists(path_to_output):
            os.mkdir(path_to_output)
        print(class_name)

        # List the original audio files in the class directory
        original_audio_list = os.listdir(this_audio_path)
        number2 = 0
        for audio in original_audio_list:
            number2 += 1
            if number2 > 10:
                break
            min_metric = 999999999
            PATH = os.path.join(this_audio_path, audio)
            wav1, sr1 = librosa.load(PATH, sr=None)
            mel_spect_1 = feature_melspect(wav1, sr1, n_fft=int(fft_ms * sr1 / 1000), 
                                           hop_length=int(hop_ms * sr1 / 1000))
            spec_len = len(mel_spect_1[0, :])
            time_len = len(wav1) / sr1
            
            # Determine the window length based on the class
            if number == 0:
                window_length = int((6000 - fft_ms) / hop_ms + 2 + spec_len)
                number = number + 1
            else:
                if class_name in class_list:
                    window_length = int((2000 - fft_ms) / hop_ms + 2 + spec_len)
                else:
                    window_length = int((1000 - fft_ms) / hop_ms + 2 + spec_len)
            
            # Find the best matching position for the watermark
            for n in range(window_length - spec_len):
                mse_spect = mse(mel_spect_0[:, n + pos:n + spec_len + pos], 
                                       mel_spect_1)
                if min_metric > mse_spect:
                    min_metric = mse_spect
                    pos_temp = n
            pos = pos + pos_temp
            audio_pos = pos * int(hop_ms * sr / 1000)
            audio_len = int(time_len * sr)
            
            # Add a small margin to the extracted audio segment
            audio1 = wav[int(audio_pos - 0.2 * sr):int(audio_pos + audio_len + 0.2 * sr)]
            pos = pos + spec_len

            # Write out audio as 16-bit PCM WAV file
            sf.write(os.path.join(path_to_output, audio), audio1, sr, subtype='PCM_16')
