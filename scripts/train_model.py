import os
import pickle
from extract_features import extract_mfcc

def train_templates():
    templates = {}
    base_path = './data/templates/'

    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        templates[person] = {}

        for vowel in os.listdir(person_path):
            file_path = os.path.join(person_path, vowel)
            mfcc_features = extract_mfcc(file_path)
            templates[person][vowel.split('.')[0]] = mfcc_features

    with open('templates.pkl', 'wb') as f:
        pickle.dump(templates, f)

if __name__ == "__main__":
    train_templates()
