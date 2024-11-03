from test_model import test_model

def calculate_accuracy(results):
    correct = 0
    total = 0
    for person, vowels in results.items():
        for vowel, predicted_vowel in vowels.items():
            if vowel[0] == predicted_vowel[0]:
                correct += 1
            total += 1
    return correct / total * 100

if __name__ == "__main__":
    results = test_model()
    print(results)
    accuracy = calculate_accuracy(results)
    print(f"Accuracy: {accuracy:.2f}%")
