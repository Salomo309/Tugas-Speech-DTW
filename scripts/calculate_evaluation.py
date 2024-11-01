from test_model import test_model

def calculate_accuracy(results):
    correct = 0
    total = 0
    for person, vowels in results.items():
        for vowel, (predicted_person, predicted_vowel) in vowels.items():
            if vowel == predicted_vowel:
                correct += 1
            total += 1
    return correct / total * 100

if __name__ == "__main__":
    results = test_model()
    accuracy = calculate_accuracy(results)
    print(f"Accuracy: {accuracy:.2f}%")
