"""
Spam Message Identifier using Machine Learning
University of Eastern Africa Baraton
Authors: Dennis Kipkemoi, Baraka Kahindi, Lenard Kibet
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class SpamClassifier:
    """
    A machine learning-based spam message classifier using TF-IDF and Multinomial Naive Bayes
    """

    def __init__(self):
        # We clean text manually first, then vectorize cleaned text.
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
        self.classifier = MultinomialNB()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.is_trained = False

    @staticmethod
    def _normalize_column_name(column_name):
        """Normalize a column name to make matching easier."""
        return str(column_name).strip().lower().replace(' ', '').replace('_', '')

    def _detect_columns(self, df):
        """Detect label and message columns from common dataset naming conventions."""
        normalized = {self._normalize_column_name(col): col for col in df.columns}

        label_candidates = ['label', 'class', 'category', 'target', 'type', 'v1']
        message_candidates = ['message', 'text', 'sms', 'content', 'body', 'v2']

        label_col = next((normalized[c] for c in label_candidates if c in normalized), None)
        message_col = next((normalized[c] for c in message_candidates if c in normalized), None)

        # Fallback: use the first two columns if explicit names are missing.
        if label_col is None or message_col is None:
            if len(df.columns) < 2:
                raise ValueError("Dataset must contain at least two columns for label and message.")
            label_col = df.columns[0]
            message_col = df.columns[1]

        return label_col, message_col

    def _convert_labels_to_binary(self, label_series):
        """Convert common spam/ham labels to 1/0."""
        # First, try direct numeric conversion for datasets that already use 0/1.
        numeric_labels = pd.to_numeric(label_series, errors='coerce')
        if numeric_labels.notna().all() and set(numeric_labels.unique()).issubset({0, 1}):
            return numeric_labels.astype(int)

        # Otherwise, map text labels to binary values.
        normalized = label_series.astype(str).str.strip().str.lower()
        mapping = {
            'spam': 1,
            'ham': 0,
            'not spam': 0,
            'legitimate': 0,
            'non-spam': 0,
            'nonspam': 0,
            '0': 0,
            '1': 1
        }
        converted = normalized.map(mapping)

        if converted.isna().any():
            unknown_values = sorted(normalized[converted.isna()].unique())
            raise ValueError(
                f"Unsupported label values found: {unknown_values}. "
                "Please use labels like spam/ham or 1/0."
            )

        return converted.astype(int)

    def clean_text(self, text):
        """
        Clean one message:
        - lowercase
        - remove punctuation
        - remove stopwords
        """
        if pd.isna(text):
            return ''

        text = str(text).lower()

        # Remove punctuation and symbols, keep letters/numbers/spaces.
        text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)

        # Collapse repeated spaces.
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stop words.
        tokens = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)

    def load_data(self, filepath, label_col=None, message_col=None):
        """
        Load spam dataset from CSV/TSV and standardize to:
        - label: 1 for spam, 0 for ham
        - message: raw text message

        If column names differ, they are auto-detected.
        """
        try:
            # Try CSV/TSV with automatic delimiter detection.
            try:
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8')
                except Exception:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin-1')

                # If parsing produced only one column, fallback to tab-separated format.
                if len(df.columns) < 2:
                    raise ValueError("Auto delimiter detection produced fewer than 2 columns")
            except Exception:
                # Common format for SMS Spam Collection: TSV with no header.
                try:
                    df = pd.read_csv(filepath, sep='\t', header=None, encoding='utf-8')
                except Exception:
                    df = pd.read_csv(filepath, sep='\t', header=None, encoding='latin-1')

            # Some Kaggle datasets include unnamed index columns. Drop fully empty columns.
            df = df.dropna(axis=1, how='all')

            if label_col is None or message_col is None:
                detected_label, detected_message = self._detect_columns(df)
                label_col = label_col or detected_label
                message_col = message_col or detected_message

            # Keep only needed columns and rename to a standard schema.
            df = df[[label_col, message_col]].copy()
            df.columns = ['label', 'message']

            # Remove rows with missing values.
            df = df.dropna(subset=['label', 'message'])

            # Convert labels to binary values.
            df['label'] = self._convert_labels_to_binary(df['label'])

            # Ensure message column is text.
            df['message'] = df['message'].astype(str)

            print(f"✓ Dataset loaded successfully!")
            print(f"  Total messages: {len(df)}")
            print(f"  Spam messages: {(df['label'] == 1).sum()}")
            print(f"  Ham messages: {(df['label'] == 0).sum()}")
            print(f"  Label column used: {label_col}")
            print(f"  Message column used: {message_col}")

            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None

    def preprocess_data(self, df):
        """
        Preprocess the dataset
        - lowercase
        - remove punctuation
        - remove stopwords
        """
        cleaned_df = df.copy()
        cleaned_df['message'] = cleaned_df['message'].apply(self.clean_text)

        X = cleaned_df['message']
        y = df['label']

        print("\n✓ Preprocessing data...")
        print("  - Converting to lowercase")
        print("  - Removing punctuation and special characters")
        print("  - Removing stop words")
        print("  - Text cleaning complete")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets (80/20 split)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n✓ Data split completed (80/20):")
        print(f"  Training set: {len(X_train)} messages")
        print(f"  Testing set: {len(X_test)} messages")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the Multinomial Naive Bayes classifier
        """
        print("\n✓ Training the Multinomial Naive Bayes classifier...")

        # Convert cleaned text to TF-IDF features.
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train the classifier
        self.classifier.fit(X_train_vec, y_train)

        print(f"  Training completed!")
        print(f"  Features extracted: {X_train_vec.shape[1]}")

        self.is_trained = True

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using various metrics
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None

        # Vectorize test data
        X_test_vec = self.vectorizer.transform(X_test)

        # Make predictions
        y_pred = self.classifier.predict(X_test_vec)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {conf_matrix[0][0]:5d} | False Positives: {conf_matrix[0][1]:5d}")
        print(f"  False Negatives: {conf_matrix[1][0]:5d} | True Positives:  {conf_matrix[1][1]:5d}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        print("="*60)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on the training data
        """
        print(f"\n✓ Performing {cv}-fold cross-validation...")

        X_train_vec = self.vectorizer.fit_transform(X_train)

        scores = cross_val_score(self.classifier, X_train_vec, y_train, cv=cv, scoring='accuracy')

        print(f"  Cross-validation scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return scores

    def predict(self, message):
        """
        Classify a single message as spam or ham
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None

        cleaned_message = self.clean_text(message)
        message_vec = self.vectorizer.transform([cleaned_message])
        prediction = self.classifier.predict(message_vec)[0]
        probability = self.classifier.predict_proba(message_vec)[0]

        label = "SPAM" if prediction == 1 else "HAM"
        spam_probability = probability[1]
        ham_probability = probability[0]

        return {
            'label': label,
            'spam_probability': spam_probability,
            'ham_probability': ham_probability
        }

    def predict_batch(self, messages):
        """
        Classify multiple messages
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None

        results = []
        for message in messages:
            result = self.predict(message)
            results.append(result)

        return results

    def save_model(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Save the trained model and vectorizer to files
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"\n✓ Model saved successfully!")
            print(f"  Model file: {model_path}")
            print(f"  Vectorizer file: {vectorizer_path}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")

    def load_model(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Load a previously trained model and vectorizer
        """
        try:
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            print(f"✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")


def main():
    """
    Main function to demonstrate the spam classifier
    """
    print("\n" + "="*60)
    print("SPAM MESSAGE IDENTIFIER - MACHINE LEARNING")
    print("University of Eastern Africa Baraton")
    print("="*60)

    # Initialize classifier
    classifier = SpamClassifier()

    # Load dataset
    print("\n[1/9] Loading Dataset...")
    dataset_path = input("Enter dataset path (press Enter for 'spam.csv'): ").strip() or 'spam.csv'
    df = classifier.load_data(dataset_path)

    if df is None:
        print("\n✗ Failed to load dataset.")
        print("  Ensure the CSV/TSV file exists and contains label/text columns.")
        return

    # Preprocess data
    print("\n[2/9] Preprocessing Data...")
    X, y = classifier.preprocess_data(df)

    # Split data
    print("\n[3/9] Splitting Data...")
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    # Cross-validation
    print("\n[4/9] Cross-Validation...")
    classifier.cross_validate(X_train, y_train, cv=5)

    # Train model
    print("\n[5/9] Training Model...")
    classifier.train(X_train, y_train)

    # Evaluate model
    classifier.evaluate(X_test, y_test)

    # Save model
    classifier.save_model()

    # Test with sample messages
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)

    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, how are you doing? Let's catch up soon!",
        "URGENT: Your account has been compromised. Verify identity now.",
        "Hi John, meeting at 3pm tomorrow?",
        "You are a winner! Claim your prize of £1000000",
        "Thanks for your purchase. Order confirmation attached."
    ]

    for msg in test_messages:
        result = classifier.predict(msg)
        confidence = max(result['spam_probability'], result['ham_probability'])
        print(f"\nMessage: {msg[:50]}...")
        print(f"  Classification: {result['label']}")
        print(f"  Spam Probability: {result['spam_probability']:.2%}")
        print(f"  Confidence: {confidence:.2%}")

    # Interactive mode for user input.
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION")
    print("="*60)
    print("Type your own message and press Enter.")
    print("Type 'quit' to stop.")

    while True:
        user_message = input("\nEnter message: ").strip()

        if user_message.lower() in {'quit', 'exit'}:
            print("Exiting interactive mode.")
            break

        if not user_message:
            print("Please enter a non-empty message.")
            continue

        result = classifier.predict(user_message)
        print(f"Classification: {result['label']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Ham Probability: {result['ham_probability']:.2%}")


if __name__ == "__main__":
    main()
