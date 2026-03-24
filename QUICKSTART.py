"""
Quick Start Guide - Get the Spam Classifier Running in Minutes!
"""

def print_quickstart():
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║          SPAM MESSAGE IDENTIFIER - QUICK START GUIDE                 ║
║                     (Python Machine Learning)                        ║
╚══════════════════════════════════════════════════════════════════════╝

🚀 STEP 1: INSTALL DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install -r requirements.txt

    Expected output: Successfully installed scikit-learn, pandas, numpy


📊 STEP 2: DOWNLOAD THE DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    2. Download the "spam.csv" file
    3. Place it in the project folder


🤖 STEP 3: TRAIN THE MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python spam_classifier.py

    This will:
    ✓ Load the SMS dataset (5,574 messages)
    ✓ Preprocess the data
    ✓ Train the model
    ✓ Evaluate performance (should see ~98% accuracy)
    ✓ Save trained model files


✨ STEP 4: TEST THE MODEL (Interactive Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python demo.py

    Then select "1" for interactive mode and type messages to classify!


📋 STEP 5: EXPLORE ADVANCED FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python examples.py

    Try different examples:
    - Basic predictions
    - Batch processing
    - Confidence filtering
    - Statistics and analytics
    - JSON export


═══════════════════════════════════════════════════════════════════════

📁 PROJECT FILES EXPLAINED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  spam_classifier.py    → Main classifier class (train & evaluate)
  demo.py              → Interactive testing interface
  examples.py          → Advanced usage examples
  requirements.txt     → Python dependencies
  README.md            → Comprehensive documentation
  spam.csv             → Dataset (download separately)
  spam_model.pkl       → Trained model (generated)
  vectorizer.pkl       → Text vectorizer (generated)


═══════════════════════════════════════════════════════════════════════

🎯 QUICK CLASSIFICATION EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from spam_classifier import SpamClassifier

    classifier = SpamClassifier()
    classifier.load_model()

    result = classifier.predict("Free iPhone! Click here!")
    print(result['label'])  # Output: SPAM


═══════════════════════════════════════════════════════════════════════

📊 EXPECTED PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ✓ Accuracy:  ~98.5%
    ✓ Precision: ~98.9%
    ✓ Recall:    ~95.4%
    ✓ F1-Score:  ~97.1%


═══════════════════════════════════════════════════════════════════════

🔧 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ❌ FileNotFoundError: spam.csv not found
     → Download the dataset from Kaggle

  ❌ ModuleNotFoundError: No module named 'sklearn'
     → Run: pip install -r requirements.txt

  ❌ Model file not found (when running demo.py)
     → First run: python spam_classifier.py


═══════════════════════════════════════════════════════════════════════

💡 TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • The model needs the dataset to train (spam.csv)
  • Training takes ~30-60 seconds depending on your computer
  • The trained model is saved for future use
  • You can use the model without retraining via demo.py
  • Read README.md for detailed information


═══════════════════════════════════════════════════════════════════════

Need help? Check README.md for comprehensive documentation!

"""
    print(guide)


if __name__ == "__main__":
    print_quickstart()
