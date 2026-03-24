"""
Unit Tests for Spam Message Classifier
Run with: python -m pytest test_spam_classifier.py -v
"""

import unittest
import pandas as pd
import numpy as np
from spam_classifier import SpamClassifier
import os
import tempfile


class TestSpamClassifier(unittest.TestCase):
    """Test suite for SpamClassifier class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.classifier = SpamClassifier()

    def test_classifier_initialization(self):
        """Test that classifier initializes correctly"""
        self.assertIsNotNone(self.classifier)
        self.assertFalse(self.classifier.is_trained)
        self.assertIsNotNone(self.classifier.vectorizer)
        self.assertIsNotNone(self.classifier.classifier)

    def test_tfidf_vectorizer_config(self):
        """Test TF-IDF vectorizer configuration"""
        self.assertEqual(self.classifier.vectorizer.max_features, 5000)
        self.assertEqual(self.classifier.vectorizer.lowercase, True)
        self.assertIsNotNone(self.classifier.vectorizer.stop_words)

    def test_create_sample_dataset(self):
        """Test creating a sample dataset"""
        # Create sample data
        data = {
            'label': ['spam', 'ham', 'spam', 'ham', 'spam'],
            'message': [
                'Free money now!',
                'Hey how are you?',
                'Click here for prizes',
                'See you tomorrow',
                'Win free gift card'
            ]
        }
        df = pd.DataFrame(data)

        # Convert labels
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})

        self.assertEqual(len(df), 5)
        self.assertEqual(df[df['label'] == 1].shape[0], 3)  # 3 spam messages
        self.assertEqual(df[df['label'] == 0].shape[0], 2)  # 2 ham messages

    def test_preprocessing(self):
        """Test data preprocessing"""
        data = {
            'label': ['spam', 'ham'],
            'message': ['Free Prize!!!', 'Hi there']
        }
        df = pd.DataFrame(data)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})

        X, y = self.classifier.preprocess_data(df)

        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)

    def test_data_splitting(self):
        """Test train/test split"""
        data = {
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'message': [
                'message 1', 'message 2', 'message 3', 'message 4',
                'message 5', 'message 6', 'message 7', 'message 8',
                'message 9', 'message 10'
            ]
        }
        df = pd.DataFrame(data)
        X = df['message']
        y = df['label']

        X_train, X_test, y_train, y_test = self.classifier.split_data(X, y, test_size=0.2, random_state=42)

        # Check sizes (80/20 split)
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(y_test), 2)

    def test_model_training(self):
        """Test model training"""
        # Create simple training data
        messages = [
            'Free prize now',
            'Click here',
            'Hi how are you',
            'See you later',
            'Limited offer',
            'Call me soon'
        ]
        labels = np.array([1, 1, 0, 0, 1, 0])

        # Train classifier
        self.classifier.train(messages, labels)

        # Check that model is trained
        self.assertTrue(self.classifier.is_trained)

    def test_single_prediction_spam(self):
        """Test prediction on a spam message"""
        # First train the classifier
        messages = [
            'Free prize', 'Click here', 'Win money',
            'Hello friend', 'How are you', 'See you'
        ]
        labels = np.array([1, 1, 1, 0, 0, 0])
        self.classifier.train(messages, labels)

        # Test spam prediction
        result = self.classifier.predict('Free money now!')

        self.assertIsNotNone(result)
        self.assertIn('label', result)
        self.assertIn('spam_probability', result)
        self.assertIn('ham_probability', result)
        self.assertEqual(result['label'], 'SPAM')
        self.assertGreater(result['spam_probability'], 0.5)

    def test_single_prediction_ham(self):
        """Test prediction on a legitimate message"""
        # First train the classifier
        messages = [
            'Free prize', 'Click here', 'Win money',
            'Hello friend', 'How are you', 'See you'
        ]
        labels = np.array([1, 1, 1, 0, 0, 0])
        self.classifier.train(messages, labels)

        # Test ham prediction
        result = self.classifier.predict('How are you today?')

        self.assertIsNotNone(result)
        self.assertEqual(result['label'], 'HAM')
        self.assertLess(result['spam_probability'], 0.5)

    def test_batch_prediction(self):
        """Test batch predictions"""
        # First train the classifier
        messages = [
            'Free prize', 'Click here', 'Win money',
            'Hello friend', 'How are you', 'See you'
        ]
        labels = np.array([1, 1, 1, 0, 0, 0])
        self.classifier.train(messages, labels)

        # Test batch prediction
        test_messages = ['Free money!', 'Hi there', 'Win now!']
        results = self.classifier.predict_batch(test_messages)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('label', result)
            self.assertIn('spam_probability', result)
            self.assertIn('ham_probability', result)

    def test_probability_sum_to_one(self):
        """Test that spam and ham probabilities sum to 1"""
        # First train the classifier
        messages = [
            'Free prize', 'Click here', 'Win money',
            'Hello friend', 'How are you', 'See you'
        ]
        labels = np.array([1, 1, 1, 0, 0, 0])
        self.classifier.train(messages, labels)

        result = self.classifier.predict('Test message')

        # Check probabilities sum to approximately 1
        total_prob = result['spam_probability'] + result['ham_probability']
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_model_save_and_load(self):
        """Test saving and loading the model"""
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            vectorizer_path = os.path.join(tmpdir, 'test_vectorizer.pkl')

            # Train and save
            messages = [
                'Free prize', 'Click here', 'Win money',
                'Hello friend', 'How are you', 'See you'
            ]
            labels = np.array([1, 1, 1, 0, 0, 0])
            self.classifier.train(messages, labels)
            self.classifier.save_model(model_path, vectorizer_path)

            # Check files exist
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(vectorizer_path))

            # Load into new classifier
            new_classifier = SpamClassifier()
            new_classifier.load_model(model_path, vectorizer_path)

            # Test that loaded classifier works
            result = new_classifier.predict('Free money!')
            self.assertIsNotNone(result)
            self.assertIn('label', result)

    def test_empty_message_handling(self):
        """Test handling of empty messages"""
        # Train classifier first
        messages = ['Free prize', 'Hello there']
        labels = np.array([1, 0])
        self.classifier.train(messages, labels)

        # Empty string should work but likely be classified as ham
        # (depends on the model, but should not crash)
        try:
            result = self.classifier.predict('')
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Empty message handling failed: {e}")

    def test_long_message_handling(self):
        """Test handling of very long messages"""
        # Train classifier first
        messages = ['Free prize', 'Hello there']
        labels = np.array([1, 0])
        self.classifier.train(messages, labels)

        # Very long message
        long_message = 'a ' * 1000
        try:
            result = self.classifier.predict(long_message)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Long message handling failed: {e}")

    def test_special_characters_handling(self):
        """Test handling of special characters"""
        # Train classifier first
        messages = ['Free prize!', 'Hello @friend']
        labels = np.array([1, 0])
        self.classifier.train(messages, labels)

        # Message with special characters
        special_message = 'Free!!! Prize $$$ Win @@@'
        try:
            result = self.classifier.predict(special_message)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Special characters handling failed: {e}")

    def test_unicode_characters_handling(self):
        """Test handling of unicode characters"""
        # Train classifier first
        messages = ['Free prize', 'Hello friend']
        labels = np.array([1, 0])
        self.classifier.train(messages, labels)

        # Message with unicode characters
        unicode_message = 'Hello world 你好 مرحبا'
        try:
            result = self.classifier.predict(unicode_message)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Unicode handling failed: {e}")

    def test_case_insensitivity(self):
        """Test that predictions are case-insensitive"""
        # Train classifier first
        messages = ['free prize', 'hello friend']
        labels = np.array([1, 0])
        self.classifier.train(messages, labels)

        # Same message in different cases
        result1 = self.classifier.predict('FREE PRIZE')
        result2 = self.classifier.predict('free prize')
        result3 = self.classifier.predict('Free Prize')

        # Results should be very similar
        self.assertAlmostEqual(result1['spam_probability'], result2['spam_probability'], places=4)
        self.assertAlmostEqual(result1['spam_probability'], result3['spam_probability'], places=4)

    def test_prediction_without_training(self):
        """Test that prediction fails gracefully without training"""
        new_classifier = SpamClassifier()

        # Should return None or raise an appropriate error
        try:
            result = new_classifier.predict('test message')
            # If it doesn't raise, result should be None
            if result is not None:
                self.fail("Should return None or raise error when not trained")
        except Exception:
            # Expected behavior
            pass


class TestSpamClassifierIntegration(unittest.TestCase):
    """Integration tests for the classifier"""

    def setUp(self):
        """Set up for each test"""
        self.classifier = SpamClassifier()

    def test_full_workflow(self):
        """Test complete workflow: train, evaluate, predict"""
        # Create sample dataset
        data = {
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
            'message': [
                'Free money offer',
                'Hey how are you',
                'Win free prizes',
                'See you tomorrow',
                'Claim your reward',
                'Hello friend',
                'Click here now',
                'Call me later'
            ]
        }
        df = pd.DataFrame(data)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})

        # Preprocess
        X, y = self.classifier.preprocess_data(df)

        # Split
        X_train, X_test, y_train, y_test = self.classifier.split_data(X, y, test_size=0.25)

        # Train
        self.classifier.train(X_train, y_train)

        # Predict
        result = self.classifier.predict('Free money now!')
        self.assertIsNotNone(result)
        self.assertIn('label', result)


class TestDataPreprocessing(unittest.TestCase):
    """Tests for data preprocessing functionality"""

    def test_label_conversion(self):
        """Test label conversion from string to binary"""
        data = {
            'label': ['spam', 'ham', 'spam'],
            'message': ['msg1', 'msg2', 'msg3']
        }
        df = pd.DataFrame(data)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})

        self.assertEqual(list(df['label']), [1, 0, 1])

    def test_message_extraction(self):
        """Test message extraction from dataframe"""
        data = {
            'label': ['spam', 'ham'],
            'message': ['message 1', 'message 2']
        }
        df = pd.DataFrame(data)
        X = df['message']
        y = df['label']

        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)


if __name__ == '__main__':
    unittest.main()
