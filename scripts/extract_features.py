# ############################# Tugas 1 ############################# #

# import numpy as np
# import python_speech_features as psf
# import scipy.io.wavfile as wav
# from pydub import AudioSegment
# from scipy.signal import butter, filtfilt

# def apply_filter(samples, filter_type, cutoff, samplerate=16000, order=5):
#     nyquist = 0.5 * samplerate
#     normalized_cutoff = cutoff / nyquist
#     b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
#     filtered_samples = filtfilt(b, a, samples)
#     return filtered_samples

# def extract_mfcc(file_path, num_mfcc=39, use_filter=False, filter_type=None, cutoff=None):
#     audio = AudioSegment.from_file(file_path)
#     audio = audio.set_frame_rate(16000)
#     audio = audio.set_channels(1)
#     samples = np.array(audio.get_array_of_samples())

#     if use_filter and filter_type and cutoff:
#         samples = apply_filter(samples, filter_type, cutoff, samplerate=16000)

#     mfcc_features = psf.mfcc(samples, samplerate=16000, numcep=num_mfcc)
#     return mfcc_features

# ############################# Tugas 2 ############################# #
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=39):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Ekstraksi MFCC
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Transpose untuk mendapatkan dimensi (frame x n_mfcc)
    return mfcc_features.T
