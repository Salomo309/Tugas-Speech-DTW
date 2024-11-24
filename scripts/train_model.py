import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import python_speech_features as psf
from scipy.signal import butter, filtfilt
from extract_features import extract_mfcc

def process_audio_file(file_path):
    features_original = extract_mfcc(file_path, use_filter=False)
    
    features_lowpass = extract_mfcc(file_path, 
                                  use_filter=True, 
                                  filter_type='low', 
                                  cutoff=1000)
    
    features_highpass = extract_mfcc(file_path, 
                                   use_filter=True, 
                                   filter_type='high', 
                                   cutoff=500)
    
    return {
        'original': features_original,
        'lowpass': features_lowpass,
        'highpass': features_highpass
    }

def train_templates():
    templates = {
        'original': {},
        'lowpass': {},
        'highpass': {}
    }
    
    base_path = '../data/templates'
    for audio_file in os.listdir(base_path):
        if audio_file.endswith(('.wav', '.aac')):
            vowel_label = audio_file.split('.')[0]
            file_path = os.path.join(base_path, audio_file)
            
            try:
                features = process_audio_file(file_path)
                
                templates['original'][vowel_label] = features['original']
                templates['lowpass'][vowel_label] = features['lowpass']
                templates['highpass'][vowel_label] = features['highpass']
                
                print(f"Berhasil memproses file: {audio_file}")
                
            except Exception as e:
                print(f"Error memproses file {audio_file}: {str(e)}")
                continue
    
    for filter_type in templates:
        output_file = f'templates_{filter_type}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(templates[filter_type], f)
        print(f"Berhasil menyimpan template: {output_file}")

if __name__ == "__main__":
    train_templates()