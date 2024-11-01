import os
import pickle
from extract_features import extract_mfcc

def train_templates():
    templates = {}
    base_path = '../data/templates'

    for vowel_file in os.listdir(base_path):
        if vowel_file.endswith('.aac'):
            vowel_label = vowel_file.split('.')[0]
            file_path = os.path.join(base_path, vowel_file)
            mfcc_features = extract_mfcc(file_path)
            templates[vowel_label] = mfcc_features

    with open('templates.pkl', 'wb') as f:
        pickle.dump(templates, f)

if __name__ == "__main__":
    train_templates()
