import os
import pickle
from extract_features import extract_mfcc
from dtw_algorithm import dtw_distance

def test_model(use_filter=None, filter_type=None, cutoff=None):
    with open('templates.pkl', 'rb') as f:
        templates = pickle.load(f)

    base_path = '../data/test/'
    results = {}

    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        results[person] = {}

        for vowel_file in os.listdir(person_path):
            if vowel_file.endswith(('.aac', '.wav')):
                vowel_label = vowel_file.split('.')[0]

                # Extract MFCC with optional filtering
                test_mfcc = extract_mfcc(
                    os.path.join(person_path, vowel_file),
                    apply_filter=use_filter,
                    filter_type=filter_type,
                    cutoff=cutoff
                )

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

    # Experiment 1: No filter
    print("Testing without filter...")
    results_no_filter = test_model(apply_filter=False)
    print("Results (No Filter):", results_no_filter)

    # Experiment 2: High-pass filter
    print("\nTesting with high-pass filter (cutoff=500 Hz)...")
    results_highpass = test_model(use_filter=True, filter_type='high', cutoff=500)
    print("Results (High-pass Filter):", results_highpass)

    # Experiment 3: Low-pass filter
    print("\nTesting with low-pass filter (cutoff=4000 Hz)...")
    results_lowpass = test_model(use_filter=True, filter_type='low', cutoff=4000)
    print("Results (Low-pass Filter):", results_lowpass)
