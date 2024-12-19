import os
import pickle
from extract_features import extract_mfcc
from dtw_algorithm import dtw_distance

def test_model():
    templates = {}
    base_path = '../data/test/'

    # Load templates sesuai format nama file
    for n_mfcc in [13, 20, 39]:
        for filter_type in ['nofilter', 'low', 'high']:
            template_file = f'templates_{n_mfcc}_{filter_type}.pkl'
            with open(template_file, 'rb') as f:
                templates[f'n_mfcc_{n_mfcc}_filter_{filter_type}'] = pickle.load(f)

    results = {key: {} for key in templates.keys()}
    
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        for key in results:
            results[key][person] = {}
        
        for vowel_file in os.listdir(person_path):
            if vowel_file.endswith(('.aac', '.wav')):
                vowel_label = vowel_file.split('.')[0]
                file_path = os.path.join(person_path, vowel_file)
                
                try:
                    for key in templates.keys():
                        n_mfcc = int(key.split('_')[2])
                        filter_type = key.split('_')[-1]
                        
                        # Tentukan filter
                        if filter_type == 'nofilter':
                            test_mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc)
                        elif filter_type == 'low':
                            test_mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, filter_type='low', cutoff=1000)
                        elif filter_type == 'high':
                            test_mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, filter_type='high', cutoff=500)
                        
                        # Cari template terbaik
                        best_match = find_best_match(test_mfcc, templates[key])
                        results[key][person][vowel_label] = best_match
                    
                    print(f"Processed {person}/{vowel_file}")
                except Exception as e:
                    print(f"Error processing {person}/{vowel_file}: {str(e)}")
                    continue

    return results

def find_best_match(test_mfcc, templates):
    best_match = None
    min_distance = float('inf')
    
    for template_vowel, template_mfcc in templates.items():
        distance = dtw_distance(test_mfcc, template_mfcc)
        if distance < min_distance:
            min_distance = distance
            best_match = template_vowel
    
    return best_match

def calculate_accuracy(results):
    accuracies = {}
    
    for key, filter_results in results.items():
        total = 0
        correct = 0
        
        for person, person_results in filter_results.items():
            for vowel_label, predicted in person_results.items():
                total += 1
                if vowel_label[0] == predicted[0]:
                    correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracies[key] = accuracy
    
    return accuracies

if __name__ == "__main__":
    print("Starting testing process...")
    
    try:
        results = test_model()
        
        accuracies = calculate_accuracy(results)
        
        print("\nDetailed Results:")
        for key, filter_results in results.items():
            print(f"\n{key.upper()} RESULTS:")
            for person, person_results in filter_results.items():
                print(f"\nPerson: {person}")
                for vowel_label, predicted in person_results.items():
                    print(f"True: {vowel_label}, Predicted: {predicted[0]}")
        
        print("\nAccuracies:")
        for key, accuracy in accuracies.items():
            print(f"{key}: {accuracy:.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
