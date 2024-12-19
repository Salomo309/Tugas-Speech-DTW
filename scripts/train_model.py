# ############################# Tugas 2 ############################# #
import os
import pickle
import numpy as np
from extract_features import extract_mfcc
from dtw_algorithm import compute_dtw

# def pad_or_truncate_mfcc(mfcc_list, target_length):
#     processed_list = []
#     for mfcc in mfcc_list:
#         if mfcc.shape[0] > target_length:  # Truncate
#             processed_list.append(mfcc[:target_length, :])
#         else:  # Pad
#             pad_width = target_length - mfcc.shape[0]
#             padded_mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
#             processed_list.append(padded_mfcc)
#     return processed_list

def align_mfcc(reference, target):
    _, alignment_path = compute_dtw(reference, target)

    aligned_mfcc = np.zeros_like(reference)
    for i, (ref_idx, tgt_idx) in enumerate(alignment_path[:len(reference)]):
        aligned_mfcc[i] = target[tgt_idx]

    return aligned_mfcc

def calculate_average_mfcc(mfcc_list):
    # target_length = max(mfcc.shape[0] for mfcc in mfcc_list)
    # mfcc_list = pad_or_truncate_mfcc(mfcc_list, target_length)
    # return np.mean(mfcc_list, axis=0)

    reference = mfcc_list[0]

    aligned_mfccs = [reference]
    for mfcc in mfcc_list[1:]:
        aligned_mfccs.append(align_mfcc(reference, mfcc))

    return np.mean(aligned_mfccs, axis=0)

def train_templates():
    templates = {}
    base_path = '../data/templates/'

    for vowel in os.listdir(base_path):
        vowel_path = os.path.join(base_path, vowel)
        if os.path.isdir(vowel_path):
            mfcc_list = []

            for template_file in os.listdir(vowel_path):
                if template_file.endswith('.aac') or template_file.endswith('.wav'):
                    file_path = os.path.join(vowel_path, template_file)
                    mfcc_features = extract_mfcc(file_path, 39)
                    mfcc_list.append(mfcc_features)

            if mfcc_list:
                averaged_mfcc = calculate_average_mfcc(mfcc_list)
                templates[vowel] = averaged_mfcc

    # Simpan hasil template rata-rata
    with open('templates_39_nofilter.pkl', 'wb') as f:
        pickle.dump(templates, f)

if __name__ == "__main__":
    train_templates()
