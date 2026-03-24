# Contributing to Spam Message Identifier

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to principles of respect, inclusivity, and professionalism. We expect all contributors to:
- Be respectful and constructive
- Welcome diverse perspectives
- Focus on the code, not the person
- Be patient and helpful with others

## How to Contribute

### Reporting Bugs

Before creating a bug report, check the issue list to avoid duplicates.

**When reporting a bug, include:**
- Clear, descriptive title
- Step-by-step reproduction instructions
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Code snippets if relevant
- Screenshots/logs if applicable

**Example:**
```
Title: Model fails to load with pickle error

Steps:
1. Train model with python spam_classifier.py
2. Try to load with demo.py
3. Get: "ModuleNotFoundError"

Expected: Model loads successfully
Actual: Error traceback...

Environment: Windows 11, Python 3.9, scikit-learn 1.0.2
```

### Suggesting Features

Feature requests should include:
- Clear description of the feature
- Use case/benefit
- Possible implementation approach
- Code examples if applicable

**Example:**
```
Title: Add support for language detection

Description:
Currently the classifier only works with English. Adding
multi-language support would allow users to classify messages
in other languages.

Use Case: Detect spam in Arabic, Spanish, French, etc.

Possible Implementation:
- Use langdetect library to detect language
- Train separate models for each language
- Or use multilingual embeddings
```

### Pull Requests

We actively welcome pull requests!

#### Before Starting
1. Fork the repository
2. Create a new branch for your feature
3. Check existing issues/PRs to avoid duplication

#### Development Process

```bash
# 1. Clone your fork
git clone https://github.com/yourusername/spam-message-identifier.git
cd spam-message-identifier

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and test
# ... make your changes ...
pytest test_spam_classifier.py -v

# 4. Commit with clear messages
git commit -m "Add feature: description of changes"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open Pull Request on GitHub
```

#### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Add comments for complex logic
- Keep lines under 100 characters

**Example:**
```python
def classify_message(message, classifier):
    """
    Classify a single message as spam or ham.

    Args:
        message (str): The message to classify
        classifier (SpamClassifier): Trained classifier instance

    Returns:
        dict: Contains 'label', 'spam_probability', 'ham_probability'

    Raises:
        ValueError: If message is empty
    """
    if not message.strip():
        raise ValueError("Message cannot be empty")

    return classifier.predict(message)
```

#### Commit Messages

Write clear, descriptive commit messages:

```
Format: [Type] Brief description

Types: feat, fix, docs, style, refactor, test, chore

Examples:
- feat: Add batch prediction API endpoint
- fix: Handle unicode characters in preprocessing
- docs: Update installation instructions
- test: Add 5 new unit tests for edge cases
- refactor: Simplify vectorizer initialization
```

#### Testing

All submissions must include tests:

```bash
# Run existing tests
pytest test_spam_classifier.py -v

# Add tests for your changes
# Create tests following the existing pattern
# Run all tests before submitting PR
```

**Test Requirements:**
- All new code must have corresponding tests
- Tests should cover normal and edge cases
- Maintain or improve code coverage
- All tests must pass

#### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issues
Fixes #123
Related to #456

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Updated documentation
- [ ] No breaking changes
```

### Documentation

Documentation improvements are always welcome!

**Areas to improve:**
- Typo fixes
- Clearer explanations
- Better examples
- Doctring improvements
- README enhancements

## Areas for Contribution

### High Priority
- [ ] Support for additional languages
- [ ] Neural network-based classifiers
- [ ] Real-time streaming predictions
- [ ] Mobile app integration

### Medium Priority
- [ ] Advanced feature engineering
- [ ] Hyperparameter optimization
- [ ] Additional evaluation metrics
- [ ] Performance benchmarking

### Low Priority
- [ ] Code cleanup
- [ ] Additional examples
- [ ] Visualization improvements
- [ ] Documentation enhancements

## Development Setup

### Prerequisites
- Python 3.7+
- pip
- git

### Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest matplotlib seaborn flask

# Download dataset
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Save as spam.csv
```

### Running Tests

```bash
# All tests
pytest test_spam_classifier.py -v

# Specific test
pytest test_spam_classifier.py::TestSpamClassifier::test_single_prediction_spam -v

# With coverage
pytest test_spam_classifier.py --cov=spam_classifier
```

### Running the Application

```bash
# Train model
python spam_classifier.py

# Run interactive demo
python demo.py

# Run web interface
python app.py

# Generate visualizations
python visualize.py
```

## Code Review Process

1. **Initial Review**: Maintainers review code quality and approach
2. **Testing**: Verify all tests pass and coverage is maintained
3. **Feedback**: Suggestions for improvements (if any)
4. **Approval**: Once approved, PR is merged
5. **Release**: Changes included in next release

## Project Structure

```
spam-message-identifier/
├── spam_classifier.py          # Core classifier
├── test_spam_classifier.py     # Unit tests
├── visualize.py                # Visualization scripts
├── app.py                      # Flask web app
├── examples.py                 # Usage examples
├── README.md                   # Main docs
└── requirements.txt            # Dependencies
```

## Performance Guidelines

When submitting performance-related changes:

1. Provide benchmarks showing improvement
2. Test on various dataset sizes
3. Include timing comparisons
4. Document any trade-offs

## Documentation Guidelines

Good documentation includes:
- Clear descriptions
- Code examples
- Parameter explanations
- Return value descriptions
- Possible exceptions

**Example docstring:**
```python
def predict(self, message):
    """
    Classify a message as spam or ham.

    This method vectorizes the input message using the fitted TF-IDF
    vectorizer and predicts its label using the trained Naive Bayes model.

    Args:
        message (str): The message to classify. Should be a single text string.

    Returns:
        dict: Prediction result containing:
            - 'label' (str): 'SPAM' or 'HAM'
            - 'spam_probability' (float): Probability of being spam (0-1)
            - 'ham_probability' (float): Probability of being ham (0-1)

    Raises:
        RuntimeError: If model has not been trained yet
        ValueError: If message is empty

    Example:
        >>> classifier = SpamClassifier()
        >>> classifier.load_model()
        >>> result = classifier.predict("Free money!")
        >>> print(result['label'])
        'SPAM'
    """
```

## Questions?

- Check existing issues and discussions
- Review documentation carefully
- Ask question issues are welcome
- Check examples for usage patterns

## Recognition

Contributors will be recognized:
- In the project README
- In commit messages
- In release notes
- As project collaborators

## Legal

By contributing, you agree that your contributions will be licensed under
the MIT License. You represent that you have the right to contribution
and that your code doesn't violate any rights.

## Thank You!

Thank you for contributing to making Spam Message Identifier better! 🙏
Your effort helps everyone detect spam more effectively.
