import os
import pickle
from extract_features import extract_mfcc
from dtw_algorithm import dtw_distance

def test_model():
    with open('templates.pkl', 'rb') as f:
        templates = pickle.load(f)

    base_path = '../data/test/'
    results = {}

    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        results[person] = {}

        for vowel_file in os.listdir(person_path):
            if vowel_file.endswith('.aac'):
                vowel_label = vowel_file.split('.')[0]
                test_mfcc = extract_mfcc(os.path.join(person_path, vowel_file))

                # Calculate DTW distance for each template and find the best match
                best_match = None
                min_distance = float('inf')

                for template_vowel, template_mfcc in templates.items():
                    distance = dtw_distance(test_mfcc, template_mfcc)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = template_vowel

                results[person][vowel_label] = best_match
    return results

if __name__ == "__main__":
    print(test_model())
