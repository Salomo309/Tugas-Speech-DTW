import os
import pickle
from extract_features import extract_mfcc
from dtw_algorithm import dtw_distance

def test_model():
    templates = {}
    for filter_type in ['original', 'lowpass', 'highpass']:
        template_file = f'templates_{filter_type}.pkl'
        with open(template_file, 'rb') as f:
            templates[filter_type] = pickle.load(f)
    
    base_path = '../data/test/'
    results = {
        'original': {},
        'lowpass': {},
        'highpass': {}
    }
    
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        
        for filter_type in results:
            results[filter_type][person] = {}
        
        for vowel_file in os.listdir(person_path):
            if vowel_file.endswith(('.aac', '.wav')):
                vowel_label = vowel_file.split('.')[0]
                file_path = os.path.join(person_path, vowel_file)
                
                try:
                    test_original = extract_mfcc(file_path, use_filter=False)
                    test_lowpass = extract_mfcc(file_path, use_filter=True, filter_type='low', cutoff=1000)
                    test_highpass = extract_mfcc(file_path, use_filter=True, filter_type='high', cutoff=500)
                    
                    best_match = find_best_match(test_original, templates['original'])
                    results['original'][person][vowel_label] = best_match
                    
                    best_match = find_best_match(test_lowpass, templates['lowpass'])
                    results['lowpass'][person][vowel_label] = best_match
                    
                    best_match = find_best_match(test_highpass, templates['highpass'])
                    results['highpass'][person][vowel_label] = best_match
                    
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
    
    for filter_type, filter_results in results.items():
        total = 0
        correct = 0
        
        for person, person_results in filter_results.items():
            for vowel_label, predicted in person_results.items():
                total += 1
                if vowel_label == predicted:
                    correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracies[filter_type] = accuracy
    
    return accuracies

if __name__ == "__main__":
    print("Starting testing process...")
    
    try:
        results = test_model()
        
        accuracies = calculate_accuracy(results)
        
        print("\nDetailed Results:")
        for filter_type, filter_results in results.items():
            print(f"\n{filter_type.upper()} FILTER RESULTS:")
            for person, person_results in filter_results.items():
                print(f"\nPerson: {person}")
                for vowel_label, predicted in person_results.items():
                    print(f"True: {vowel_label}, Predicted: {predicted}")
        
        print("\nAccuracies:")
        for filter_type, accuracy in accuracies.items():
            print(f"{filter_type}: {accuracy:.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
