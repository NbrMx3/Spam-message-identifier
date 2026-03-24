"""
Advanced Usage Examples for Spam Classifier
Demonstrates various use cases and configurations
"""

from spam_classifier import SpamClassifier
import json

def example_1_basic_prediction():
    """
    Example 1: Load model and make simple predictions
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Prediction")
    print("="*70)

    classifier = SpamClassifier()
    classifier.load_model()

    test_message = "Congratulations! You've won a free prize!"

    result = classifier.predict(test_message)

    print(f"\nMessage: {test_message}")
    print(f"Classification: {result['label']}")
    print(f"Spam Confidence: {result['spam_probability']:.2%}")
    print(f"Ham Confidence: {result['ham_probability']:.2%}")


def example_2_batch_prediction():
    """
    Example 2: Classify multiple messages at once
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction")
    print("="*70)

    classifier = SpamClassifier()
    classifier.load_model()

    messages = [
        "Hi, are you free tomorrow?",
        "URGENT: Claim your prize money!",
        "Meeting at 3pm in the conference room",
        "Click here to verify your account",
        "Let's grab lunch next week"
    ]

    results = classifier.predict_batch(messages)

    print("\nClassifying batch of messages...\n")
    for msg, result in zip(messages, results):
        print(f"• {msg}")
        print(f"  → {result['label']} (Spam: {result['spam_probability']:.1%})\n")


def example_3_threshold_filtering():
    """
    Example 3: Filter messages based on confidence threshold
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Confidence-based Filtering")
    print("="*70)

    classifier = SpamClassifier()
    classifier.load_model()

    messages = [
        "Free money offer!",
        "Hey, how's it going?",
        "Limited time offer - act now!",
        "What time is the meeting?",
        "Click here for exclusive deals"
    ]

    # Set confidence threshold
    spam_threshold = 0.7

    print(f"\nFiltering messages with spam threshold > {spam_threshold:.0%}\n")

    high_spam = []
    moderate_spam = []
    likely_ham = []

    for msg in messages:
        result = classifier.predict(msg)
        spam_prob = result['spam_probability']

        if spam_prob > spam_threshold:
            high_spam.append(msg)
        elif spam_prob > 0.3:
            moderate_spam.append(msg)
        else:
            likely_ham.append(msg)

    print(f"📍 LIKELY SPAM ({len(high_spam)} messages):")
    for msg in high_spam:
        print(f"   - {msg}")

    print(f"\n⚡ MODERATE SPAM ({len(moderate_spam)} messages):")
    for msg in moderate_spam:
        print(f"   - {msg}")

    print(f"\n✓ LIKELY LEGITIMATE ({len(likely_ham)} messages):")
    for msg in likely_ham:
        print(f"   - {msg}")


def example_4_statistics():
    """
    Example 4: Analyze prediction statistics
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Prediction Statistics")
    print("="*70)

    classifier = SpamClassifier()
    classifier.load_model()

    # Sample messages
    messages = [
        "Free money quick loan easy",
        "Let's meet tomorrow",
        "WINNER! Claim prize now",
        "Hi, are you available?",
        "Limited offer click now",
        "Can you help me with this?",
        "You won a free gift card!",
        "Thanks for your order",
        "Verify your account immediately",
        "See you at the office"
    ]

    results = classifier.predict_batch(messages)

    spam_count = sum(1 for r in results if r['label'] == 'SPAM')
    ham_count = sum(1 for r in results if r['label'] == 'HAM')
    avg_spam_confidence = sum(r['spam_probability'] for r in results) / len(results)

    print(f"\nAnalyzing {len(messages)} messages:\n")
    print(f"Total messages: {len(messages)}")
    print(f"Spam classified: {spam_count} ({spam_count/len(messages):.1%})")
    print(f"Ham classified: {ham_count} ({ham_count/len(messages):.1%})")
    print(f"Average spam confidence: {avg_spam_confidence:.1%}")

    # Distribution
    spam_probs = [r['spam_probability'] for r in results]
    print(f"\nSpam Confidence Distribution:")
    print(f"  Min: {min(spam_probs):.2%}")
    print(f"  Max: {max(spam_probs):.2%}")
    print(f"  Avg: {sum(spam_probs)/len(spam_probs):.2%}")


def example_5_json_export():
    """
    Example 5: Export results in JSON format
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: JSON Export")
    print("="*70)

    classifier = SpamClassifier()
    classifier.load_model()

    messages = [
        "Free prize claim now!",
        "See you tomorrow",
        "Limited offer expires today"
    ]

    results = []
    for msg in messages:
        result = classifier.predict(msg)
        results.append({
            'message': msg,
            'classification': result['label'],
            'spam_probability': round(result['spam_probability'], 4),
            'ham_probability': round(result['ham_probability'], 4)
        })

    # Export to JSON
    json_output = json.dumps({'predictions': results}, indent=2)

    print("\nJSON Format Output:\n")
    print(json_output)

    # Save to file
    with open('predictions.json', 'w') as f:
        f.write(json_output)
    print("\n✓ Saved to 'predictions.json'")


def example_6_custom_classifier():
    """
    Example 6: Train classifier on custom data
    (Requires a custom CSV file with 'label' and 'message' columns)
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Training on Custom Dataset")
    print("="*70)

    classifier = SpamClassifier()

    # Try to load custom dataset
    try:
        # Your custom dataset file
        df = classifier.load_data('custom_spam_data.csv')

        if df is not None:
            X, y = classifier.preprocess_data(df)
            X_train, X_test, y_train, y_test = classifier.split_data(X, y)

            print("\n[Training phase]")
            classifier.train(X_train, y_train)

            print("\n[Evaluation phase]")
            metrics = classifier.evaluate(X_test, y_test)

            # Save the model
            classifier.save_model('custom_spam_model.pkl', 'custom_vectorizer.pkl')

            print("\n✓ Custom model trained and saved!")
        else:
            print("\nℹ️  To use this example, create 'custom_spam_data.csv' with:")
            print("   - Column 1: 'label' (spam or ham)")
            print("   - Column 2: 'message' (the text content)")

    except Exception as e:
        print(f"\nℹ️  To use this example, create 'custom_spam_data.csv' with:")
        print("   - Column 1: 'label' (spam or ham)")
        print("   - Column 2: 'message' (the text content)")
        print(f"   Error: {e}")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("SPAM CLASSIFIER - ADVANCED EXAMPLES")
    print("="*70)
    print("\nAvailable examples:")
    print("1. Basic Prediction")
    print("2. Batch Prediction")
    print("3. Confidence-based Filtering")
    print("4. Prediction Statistics")
    print("5. JSON Export")
    print("6. Custom Dataset Training")
    print("7. Run All Examples")

    choice = input("\nSelect example (1-7): ").strip()

    if choice == "1":
        example_1_basic_prediction()
    elif choice == "2":
        example_2_batch_prediction()
    elif choice == "3":
        example_3_threshold_filtering()
    elif choice == "4":
        example_4_statistics()
    elif choice == "5":
        example_5_json_export()
    elif choice == "6":
        example_6_custom_classifier()
    elif choice == "7":
        try:
            example_1_basic_prediction()
            example_2_batch_prediction()
            example_3_threshold_filtering()
            example_4_statistics()
            example_5_json_export()
        except Exception as e:
            print(f"Error running examples: {e}")
    else:
        print("Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
