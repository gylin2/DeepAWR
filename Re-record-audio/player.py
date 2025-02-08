import os
from pydub import AudioSegment
from pydub.playback import play
import time

# Define the root directory where the audio files are located
audio_path =  "your_paths"
# List of class names (subdirectories) to process
class_list = ["fma", "zh"]

# Iterate over each class folder
for class_name in class_list:
    # Construct the path to the audio files for the current class
    this_audio_path = os.path.join(audio_path, class_name, 'wm_audio')
    # List all files in the current audio directory
    original_audio_list = os.listdir(this_audio_path)
    
    # Process each audio file in the list
    for audio in original_audio_list:
        # Construct the full file path for the current audio file
        PATH = os.path.join(this_audio_path, audio)
        
        # Check the file format and read the file accordingly
        if PATH.endswith('.wav'):
            # If the file is in WAV format, read it as a WAV file
            sound = AudioSegment.from_file(PATH, 'wav')
        elif PATH.endswith('.flac'):
            # If the file is in FLAC format, read it as a FLAC file
            sound = AudioSegment.from_file(PATH, 'flac')
        else:
            # Raise an error if the file format is not supported
            raise ValueError("Unsupported file format")
        
        # Play the audio file
        play(sound)
        # Wait for 0.2 seconds before playing the next file
        time.sleep(0.2)
