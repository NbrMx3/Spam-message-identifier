# Installation & Setup Guide

## 📋 Requirements

- **Python:** 3.7 or higher
- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 2GB (4GB recommended)
- **Disk Space:** 500MB+ (for dataset and models)

## 🔍 Check Your Python Version

```bash
python --version
# or
python3 --version
```

If Python is not installed, download from: https://www.python.org/downloads/

## 💻 Installation Steps

### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/yourusername/spam-message-identifier.git
cd spam-message-identifier
```

**Option B: Download ZIP**
```bash
# Go to GitHub repository and click "Download ZIP"
# Extract the ZIP file
# Open terminal in the extracted folder
```

### Step 2: Create Virtual Environment (Optional but Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**What this installs:**
- scikit-learn (Machine Learning)
- pandas (Data processing)
- numpy (Numerical computing)
- flask (Web framework)
- matplotlib (Plotting)
- seaborn (Advanced plots)
- pytest (Testing)

### Step 4: Download the Dataset

**Manual Download:**
1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Sign in to Kaggle (create account if needed)
3. Click "Download" button
4. Extract and place `spam.csv` in the project directory

**Expected file location:**
```
spam-message-identifier/
├── spam.csv          # <-- Place here
├── spam_classifier.py
└── ...
```

**File format:**
```
label	message
spam	Go until jurong point, crazy.. Available only in bugis n great world la e buffet
ham	Ok lar... Joking wif u oni...
```

### Step 5: Train the Model

```bash
python spam_classifier.py
```

**Expected output:**
```
======================================================================
SPAM MESSAGE IDENTIFIER - MACHINE LEARNING
======================================================================

[1/5] Loading Dataset...
✓ Dataset loaded successfully!
  Total messages: 5574
  Spam messages: 747
  Ham messages: 4827

[2/5] Preprocessing Data...
✓ Preprocessing data...
  - Converting to lowercase
  - Removing punctuation and special characters
  - Removing stop words
  - Tokenizing messages
  - Vectorizing using TF-IDF

[3/5] Splitting Data...
✓ Data split completed (80/20):
  Training set: 4459 messages
  Testing set: 1115 messages

[4/5] Cross-Validation...
✓ Performing 5-fold cross-validation...
  Cross-validation scores: ['0.9762', '0.9757', '0.9759', '0.9776', '0.9750']
  Mean accuracy: 0.9761 (+/- 0.0008)

[5/5] Training Model...
✓ Training the Multinomial Naive Bayes classifier...
  Training completed!
  Features extracted: 5000

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

======================================================================

✓ Model saved successfully!
  Model file: spam_model.pkl
  Vectorizer file: vectorizer.pkl
```

**Files created:**
- `spam_model.pkl` - Trained classifier
- `vectorizer.pkl` - Text vectorizer

### Step 6: Verify Installation

Test that everything works:

```bash
python demo.py
```

Select option 2 for batch test mode to verify the model works.

## 🌐 Web Interface Setup (Optional)

### Requirements
- Flask (already in requirements.txt)
- Modern web browser

### Running the Web Server

```bash
python app.py
```

**Output:**
```
======================================================================
SPAM MESSAGE IDENTIFIER - FLASK WEB INTERFACE
======================================================================

🚀 Starting Flask server...

📍 Open your browser and go to: http://localhost:5000

📚 API Endpoints:
   POST /api/predict          - Classify a single message
   POST /api/predict-batch    - Classify multiple messages
   GET  /api/model-info       - Get model information
   GET  /api/stats            - Get application statistics
   GET  /api/health           - Health check

======================================================================
```

**Access the application:**
1. Open web browser
2. Go to: `http://localhost:5000`
3. Classify messages using the web interface

**Stop the server:**
Press `Ctrl+C` in the terminal

## 📊 Run Tests (Optional)

```bash
# Run all tests
pytest test_spam_classifier.py -v

# Run specific test
pytest test_spam_classifier.py::TestSpamClassifier::test_single_prediction_spam -v

# Run with coverage
pytest test_spam_classifier.py --cov=spam_classifier
```

## 📈 Generate Visualizations (Optional)

```bash
python visualize.py
```

Select which visualizations to generate:
1. Confusion Matrix
2. Performance Metrics
3. Class Distribution
4. ROC Curve
5. Message Length Distribution
6. Cross-Validation Scores
7. Prediction Confidence
8. Generate All

## 🔧 Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'sklearn'`

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install scikit-learn pandas numpy flask matplotlib seaborn pytest
```

### Issue 2: `FileNotFoundError: spam.csv not found`

**Cause:** Dataset not downloaded

**Solution:**
1. Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Place `spam.csv` in project directory
3. Run: `python spam_classifier.py`

### Issue 3: `Model not loaded` when running demo.py

**Cause:** Model hasn't been trained

**Solution:**
```bash
python spam_classifier.py  # Train first
python demo.py             # Then run demo
```

### Issue 4: Port 5000 already in use

**Cause:** Another application is using port 5000

**Solution:**
Option A - Kill the process on port 5000:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>
```

Option B - Use different port:
```python
# Edit app.py
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Issue 5: `pandas.errors.EmptyDataError`

**Cause:** Dataset is corrupted or wrong format

**Solution:**
1. Re-download the dataset
2. Verify the file is not corrupted
3. Check file format (should be tab-separated)

### Issue 6: Low accuracy on your custom dataset

**Cause:** Model trained on SMS, your data is different

**Solution:**
1. Retrain the model with your dataset
2. Ensure data is in correct format (label, message columns)
3. Consider data preprocessing differences

### Issue 7: `Permission denied` errors

**Cause:** Insufficient file permissions

**Solution:**
```bash
# Windows: Run terminal as Administrator

# macOS/Linux
chmod +x spam_classifier.py
python spam_classifier.py
```

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spam.csv from Kaggle and place in this directory

# 3. Train the model
python spam_classifier.py

# 4. Run interactive demo
python demo.py

# 5. (Optional) Run web interface
python app.py

# 6. (Optional) Run tests
pytest test_spam_classifier.py -v

# 7. (Optional) Generate visualizations
python visualize.py
```

## 📦 Production Deployment

### Using Gunicorn (Production Server)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t spam-classifier .
docker run -p 5000:5000 spam-classifier
```

## 📖 Next Steps

1. **Read Documentation:** Check README.md for detailed guide
2. **Run Examples:** Execute examples.py to see different use cases
3. **Explore Code:** Review spam_classifier.py to understand implementation
4. **Try Web Interface:** Run app.py and test the web UI
5. **Customize:** Modify code for your specific needs

## 💡 Tips

- Use a virtual environment to avoid dependency conflicts
- Keep the models (pickle files) safe - they're needed for predictions
- For large-scale use, consider deployment options
- Monitor model performance over time
- Update dependencies periodically: `pip install -r requirements.txt --upgrade`

## 🆘 Getting Help

1. Check this guide again
2. Review README.md
3. Check examples.py
4. Review docstrings in source code: `pydoc spam_classifier`
5. Open an issue on GitHub with details

## ✅ Verification Checklist

After installation, verify:

- [ ] Python 3.7+ is installed
- [ ] Virtual environment is created (if using one)
- [ ] All dependencies are installed (`pip list`)
- [ ] Dataset (spam.csv) is in project directory
- [ ] Model training completes successfully
- [ ] Model achieves ~98% accuracy
- [ ] demo.py runs without errors
- [ ] Predictions work correctly

Congratulations! You've successfully set up the Spam Message Identifier! 🎉
