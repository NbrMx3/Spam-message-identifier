# Spam Message Identifier Using Machine Learning

**University of Eastern Africa Baraton**

**Authors:** Dennis Kipkemoi (SDENKI2314), Baraka Kahindi (SBRIBA2211), Lenard Kibet (SLENKI2311)

## Overview

This project implements a **Spam Message Identifier** using supervised machine learning. The system uses **TF-IDF vectorization** and **Multinomial Naive Bayes classification** to automatically identify spam messages from legitimate (ham) messages with high accuracy.

### Key Features

- ✅ **Automated Spam Detection** - Classifies messages as spam or legitimate
- ✅ **TF-IDF Vectorization** - Efficient text feature extraction
- ✅ **Multinomial Naive Bayes** - Fast and effective text classification
- ✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1-Score metrics
- ✅ **Cross-Validation** - Robust model validation
- ✅ **Model Persistence** - Save and load trained models
- ✅ **Interactive Demo** - Test with custom messages
- ✅ **Batch Testing** - Evaluate multiple messages at once

## Project Structure

```
spam-message-identifier/
├── spam_classifier.py      # Main classifier class
├── demo.py                 # Interactive demo interface
├── requirements.txt        # Python dependencies
├── spam.csv               # Dataset (download separately)
├── spam_model.pkl         # Trained model (generated)
├── vectorizer.pkl         # TF-IDF vectorizer (generated)
└── README.md              # This file
```

## Installation & Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- psycopg2-binary >= 2.9.0 (for PostgreSQL logging)

### 2. Configure PostgreSQL (Optional, for logging predictions)

Set your Neon/PostgreSQL connection string in the `DATABASE_URL` environment variable.

You can start from `.env.example` and copy the value.

Windows PowerShell:
```powershell
$env:DATABASE_URL="postgresql://username:password@host:5432/dbname?sslmode=require"
```

Then test the connection:
```bash
python test_db_connection.py
```

If successful, the app will create a table named `prediction_logs` automatically.

### 3. Download the Dataset

Download the **SMS Spam Collection Dataset** from Kaggle:

- **Dataset Link:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **File needed:** `spam.csv`
- **Place it** in the same directory as `spam_classifier.py`

**Dataset Format:**
```
label    message
spam     Free entry in 2 a wkly comp to win FA Cup...
ham      Even my brother is not like to drive...
spam     WINNER!! This is the secret code to unlock...
```

## Usage

### Option 1: Train and Evaluate the Model

```bash
python spam_classifier.py
```

**This will:**
1. Load the SMS Spam Collection dataset
2. Preprocess messages (lowercase, remove stop words, tokenize)
3. Split data (80% training, 20% testing)
4. Perform 5-fold cross-validation
5. Train Multinomial Naive Bayes classifier
6. Evaluate model performance
7. Save the trained model for later use
8. Display sample predictions

**Expected Output:**
```
============================================================
MODEL EVALUATION RESULTS
============================================================

Accuracy:  0.9850
Precision: 0.9891
Recall:    0.9535
F1-Score:  0.9710

Confusion Matrix:
  True Negatives:   966 | False Positives:  10
  False Negatives:   15 | True Positives:  240
============================================================
```

### Option 2: Run Interactive Demo

```bash
python demo.py
```

**Select Option 1** for interactive mode:

```
Select mode:
1. Interactive mode (enter custom messages)
2. Batch test mode (test predefined messages)

Enter choice (1 or 2): 1

>>> Enter a message (or 'quit' to exit): You've won a free iPhone!

  Classification: SPAM
  Spam Probability:  87.32%
  Ham Probability:   12.68%
  ⚠️  HIGH PROBABILITY OF SPAM
```

### Option 3: Batch Testing

```bash
python demo.py
```

**Select Option 2** to test predefined messages:

```
Select mode:
1. Interactive mode (enter custom messages)
2. Batch test mode (test predefined messages)

Enter choice (1 or 2): 2
```

### Option 4: Run the Flask App with DB Logging

```bash
python app.py
```

For each prediction request to `POST /api/predict`, the app attempts to save:
- message
- predicted label
- spam probability
- ham probability
- risk level
- timestamp

If DB is unavailable, prediction still works and logging is skipped gracefully.

## How It Works

### 1. **Data Preprocessing**

- Convert messages to lowercase
- Remove punctuation and special characters
- Eliminate stop words (common words like "the", "is", "a")
- Tokenize messages into words

### 2. **Vectorization (TF-IDF)**

**TF-IDF (Term Frequency - Inverse Document Frequency)** transforms text into numerical features:

- **Term Frequency (TF):** How often a word appears in a document
- **Inverse Document Frequency (IDF):** How rare a word is across all documents
- **Result:** Words that are common in spam but rare in ham get higher weights

Example:
```
Message: "Free money now!"
Vector:  [0.45, 0.32, 0.61, 0.28, ...]  (5000 features)
```

### 3. **Model Training**

**Multinomial Naive Bayes** is trained using:
- 80% of messages (training set)
- Learns probability of words given spam/ham labels
- Fast and effective for text classification

### 4. **Prediction**

For a new message:
1. Convert to TF-IDF vector
2. Calculate P(Spam|Message) and P(Ham|Message)
3. Return classification with probability scores

**Example:**
```
Message: "Congratulations you won!"
P(Spam|Message) = 0.92
P(Ham|Message) = 0.08
Classification: SPAM
```

## Performance Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Of predicted spam, how many are actually spam |
| **Recall** | TP/(TP+FN) | Of actual spam, how many are detected |
| **F1-Score** | 2*(Precision*Recall)/(Precision+Recall) | Balance between precision and recall |

**Confusion Matrix:**
```
                  Predicted Spam    Predicted Ham
Actual Spam    [True Positive]     [False Negative]
Actual Ham     [False Positive]    [True Negative]
```

## Model Files

After training, two files are created:

- **`spam_model.pkl`** - Trained Multinomial Naive Bayes classifier
- **`vectorizer.pkl`** - Fitted TF-IDF vectorizer

These files allow you to:
- Load the model without retraining
- Make predictions on new messages
- Share the model with others

## Example Usage in Code

```python
from spam_classifier import SpamClassifier

# Initialize classifier
clf = SpamClassifier()

# Load pre-trained model
clf.load_model()

# Classify a single message
result = clf.predict("You've won a free prize!")
print(result)
# Output: {'label': 'SPAM', 'spam_probability': 0.92, 'ham_probability': 0.08}

# Classify multiple messages
messages = ["Hi, how are you?", "Click here for free money!"]
results = clf.predict_batch(messages)
for result in results:
    print(result['label'])
```

## Preprocessing Details

The preprocessing pipeline handles:

1. **Case Normalization** → "HeLLo" → "hello"
2. **Stop Word Removal** → removes "the", "is", "a", "and", etc.
3. **Punctuation Removal** → "Hello!" → "Hello"
4. **Tokenization** → "Hello world" → ["hello", "world"]
5. **TF-IDF Vectorization** → ["hello", "world"] → [0.45, 0.32, ...]

## Troubleshooting

### Issue: `FileNotFoundError: spam.csv not found`

**Solution:** Download the dataset from Kaggle and place it in the project directory.

### Issue: `No module named 'sklearn'`

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Model file not found when running demo.py

**Solution:** First run `python spam_classifier.py` to train and save the model.

### Issue: Low accuracy on custom dataset

**Solution:** The model is trained on SMS messages. For email spam, consider:
- Using an email spam dataset
- Retraining the model with email-specific data
- Adjusting TF-IDF parameters (max_features, ngram_range)

## Dataset Information

**SMS Spam Collection Dataset:**
- **Total messages:** ~5,574
- **Spam messages:** ~747 (13.4%)
- **Ham messages:** ~4,827 (86.6%)
- **Language:** English
- **Source:** UCI Machine Learning Repository

**Citation:**
```
Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011).
Contributions to the Study of SMS Spam Filtering: A comparative study of
classifier performance. In 2011 IEEE Symposium on Computers and Communications (ISCC) (pp. 899-904).
```

## References

1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. Expert Systems with Applications.

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

3. Scikit-learn Developers (2024). Scikit-learn Documentation. https://scikit-learn.org/

4. UCI Machine Learning Repository. SMS Spam Collection Dataset. https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Future Improvements

- [ ] Try other classifiers (SVM, Random Forest, Neural Networks)
- [ ] Implement feature engineering (n-grams, word embeddings)
- [ ] Add support for multiple languages
- [ ] Create a web/API interface (Flask, FastAPI)
- [ ] Deploy model as a service
- [ ] Add real-time classification with streaming data
- [ ] Implement active learning for continuous improvement

## Project Team

- **Dennis Kipkemoi** (SDENKI2314)
- **Baraka Kahindi** (SBRIBA2211)
- **Lenard Kibet** (SLENKI2311)

**Institution:** University of Eastern Africa Baraton

---

**Last Updated:** March 2026
