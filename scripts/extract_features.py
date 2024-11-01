import numpy as np
import python_speech_features as psf
import scipy.io.wavfile as wav
from pydub import AudioSegment

def extract_mfcc(file_path, num_mfcc=39):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples())

    mfcc_features = psf.mfcc(samples, samplerate=16000, numcep=num_mfcc)
    return mfcc_features
