"""
Interactive Spam Message Classifier Demo
Test the trained model with custom messages
"""

from spam_classifier import SpamClassifier

def interactive_demo():
    """
    Interactive interface for testing spam classification
    """
    print("\n" + "="*70)
    print("SPAM MESSAGE IDENTIFIER - INTERACTIVE DEMO")
    print("="*70)

    classifier = SpamClassifier()

    # Try to load existing model
    try:
        classifier.load_model()
        print("\n✓ Loaded previously trained model!")
    except:
        print("\n✗ No trained model found. Please run spam_classifier.py first to train the model.")
        return

    print("\nEnter messages to classify (type 'quit' to exit):")
    print("-" * 70)

    while True:
        message = input("\n>>> Enter a message (or 'quit' to exit): ").strip()

        if message.lower() == 'quit':
            print("\n✓ Exiting classifier. Goodbye!")
            break

        if not message:
            print("✗ Please enter a valid message.")
            continue

        result = classifier.predict(message)

        print(f"\n  Classification: {result['label']}")
        print(f"  Spam Probability:  {result['spam_probability']:.2%}")
        print(f"  Ham Probability:   {result['ham_probability']:.2%}")

        if result['spam_probability'] > 0.7:
            print("  ⚠️  HIGH PROBABILITY OF SPAM")
        elif result['spam_probability'] > 0.3:
            print("  ⚡ MODERATE SPAM PROBABILITY")
        else:
            print("  ✓ Likely legitimate message")

    print("="*70)


def batch_test():
    """
    Test multiple predefined messages
    """
    print("\n" + "="*70)
    print("SPAM MESSAGE IDENTIFIER - BATCH TEST")
    print("="*70)

    classifier = SpamClassifier()

    # Try to load existing model
    try:
        classifier.load_model()
    except:
        print("\n✗ No trained model found. Please run spam_classifier.py first.")
        return

    # Predefined test messages
    test_messages = {
        "Spam_1": "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
        "Spam_2": "URGENT: Your bank account is compromised. Verify identity immediately!",
        "Spam_3": "You are a winner! Claim your prize of £1000000. Click here: bit.ly/claim",
        "Spam_4": "Free money! Make $5000 per week from home. No work required!",
        "Ham_1": "Hey, how are you doing? Let's catch up soon!",
        "Ham_2": "Hi John, meeting at 3pm tomorrow at the office.",
        "Ham_3": "Thanks for your purchase. Order confirmation attached.",
        "Ham_4": "Are we still on for dinner at 7pm Thursday?"
    }

    print("\nClassifying test messages...\n")

    results = {'spam_correct': 0, 'ham_correct': 0, 'spam_total': 0, 'ham_total': 0}

    for msg_type, message in test_messages.items():
        result = classifier.predict(message)
        category = msg_type.split('_')[0]

        # Check if classification matches expected
        is_correct = (category == "Spam" and result['label'] == "SPAM") or \
                    (category == "Ham" and result['label'] == "HAM")

        if is_correct:
            status = "✓"
            if category == "Spam":
                results['spam_correct'] += 1
            else:
                results['ham_correct'] += 1
        else:
            status = "✗"

        if category == "Spam":
            results['spam_total'] += 1
        else:
            results['ham_total'] += 1

        print(f"{status} [{msg_type}] {message[:50]}...")
        print(f"   → Classification: {result['label']} | Spam: {result['spam_probability']:.1%}\n")

    # Summary
    spam_accuracy = results['spam_correct'] / results['spam_total'] if results['spam_total'] > 0 else 0
    ham_accuracy = results['ham_correct'] / results['ham_total'] if results['ham_total'] > 0 else 0

    print("\n" + "-"*70)
    print("BATCH TEST SUMMARY")
    print("-"*70)
    print(f"Spam Classification Accuracy: {spam_accuracy:.1%} ({results['spam_correct']}/{results['spam_total']})")
    print(f"Ham Classification Accuracy:  {ham_accuracy:.1%} ({results['ham_correct']}/{results['ham_total']})")
    overall = (results['spam_correct'] + results['ham_correct']) / \
              (results['spam_total'] + results['ham_total'])
    print(f"Overall Accuracy:             {overall:.1%}")
    print("="*70)


if __name__ == "__main__":
    print("\nSelect mode:")
    print("1. Interactive mode (enter custom messages)")
    print("2. Batch test mode (test predefined messages)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        interactive_demo()
    elif choice == "2":
        batch_test()
    else:
        print("Invalid choice. Please enter 1 or 2.")
