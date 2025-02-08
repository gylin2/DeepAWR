# DeepAWR
This repository contains the official implementation of our paper, "An Audio Watermarking Method Against Re-recording Distortions".[[Paper link](https://doi.org/10.1016/j.patcog.2025.111366)]

## 1. Requirements

Installing dependencies:
```
pip install -r requirements.txt
```

### 2. Set hyperparameters and specify dataset paths
```
cd ./DeepAWQ_main
```
- ./utils/hparameter.py 

### 3. Run the training procedure
Run the `train_audio.py` to train the model on the FMA dataset:
```
python3 train_audio.py -r ./results/fma_e5_dear96 -lr_en_de 1e-5
```

### 4. Load model for watermark embedding
Run the `embed.py` to embed the watermark on the FMA dataset:
```
python3 embed.py   ./dataset/test    results/fma
```

### 5. Audio auto-recording
-Prepare a computer for playback and a mobile device for recording. Perform audio auto-recording on Windows 10 using the "Re-record-audio/player.py" tool. 
  ```
  python3 Re-record-audio/player.py
  ```
-After exporting the audio recorded on the mobile device, run the "Re-record-audio/fragment.py" tool to segment the audio and generate the re-recorded audio corresponding to the watermarked audio.
  ```
  python3 Re-record-audio/fragment.py
  ```

### 6. Load model for watermark extracting
Run the `extract.py` to extract the watermark against the re-recording distortion on the FMA dataset:
```
python3 extract.py   -2   ./results/fma/test_result/attack_audio/  ./results/fma
```
python3 extract.py   -2   ./results/fma/test_result/attack_audio_05/  ./results/fma
- **Means:** python3 extract.py &emsp; attack_method &emsp; attacked_audio_path &emsp; result_path &emsp; watermak
- attack_method: 
  - -1:&emsp;common distortion; 
  - -2:&emsp;re-recording distortion

## Citation
If you find this work useful in your research, please consider citing this paper.

## Acknowledgement
This code refers to the following project:

[1] A Deep-Learning-Based Audio Re-Recording Resilient Watermarking [[Code link](https://drive.google.com/drive/folders/1IBkXy9bBxaWsr4dneih_PYRX6ulphI1_?usp=share_link)]
