"""
Visualization Scripts for Spam Message Classifier
Visualize model performance, metrics, and analysis
Run with: python visualize.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from spam_classifier import SpamClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix():
    """
    Plot confusion matrix for the model
    """
    print("Generating confusion matrix plot...")

    # Load and prepare data
    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    except:
        print("✗ Could not load dataset. Skipping confusion matrix plot.")
        return

    # Initialize and train classifier
    classifier = SpamClassifier()
    X, y = classifier.preprocess_data(df)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    classifier.train(X_train, y_train)

    # Get predictions
    X_test_vec = classifier.vectorizer.transform(X_test)
    y_pred = classifier.classifier.predict(X_test_vec)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})

    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Spam Classifier', fontsize=14, fontweight='bold')

    # Add text annotations
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.15, f'TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}',
            ha='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.show()


def plot_performance_metrics():
    """
    Plot model performance metrics: Accuracy, Precision, Recall, F1-Score
    """
    print("Generating performance metrics plot...")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    except:
        print("✗ Could not load dataset. Skipping metrics plot.")
        return

    # Train classifier
    classifier = SpamClassifier()
    X, y = classifier.preprocess_data(df)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    classifier.train(X_train, y_train)

    # Get predictions
    X_test_vec = classifier.vectorizer.transform(X_test)
    y_pred = classifier.classifier.predict(X_test_vec)

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_metrics.png")
    plt.show()


def plot_class_distribution():
    """
    Plot distribution of spam vs ham messages in dataset
    """
    print("Generating class distribution plot...")

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        label_counts = df['label'].value_counts()
    except:
        print("✗ Could not load dataset. Skipping class distribution plot.")
        return

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = ['#27ae60', '#e74c3c']
    bars = ax1.bar(['Ham', 'Spam'], label_counts.values, color=colors, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution (Bar Chart)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    colors_pie = ['#27ae60', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(label_counts.values, labels=['Ham', 'Spam'],
                                         autopct='%1.1f%%', colors=colors_pie,
                                         startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Class Distribution (Pie Chart)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    plt.show()


def plot_roc_curve():
    """
    Plot ROC curve for the classifier
    """
    print("Generating ROC curve plot...")

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    except:
        print("✗ Could not load dataset. Skipping ROC curve plot.")
        return

    # Train classifier
    classifier = SpamClassifier()
    X, y = classifier.preprocess_data(df)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    classifier.train(X_train, y_train)

    # Get probabilities
    X_test_vec = classifier.vectorizer.transform(X_test)
    y_proba = classifier.classifier.predict_proba(X_test_vec)[:, 1]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr, tpr, color='#667eea', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Spam Classifier', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curve.png")
    plt.show()


def plot_message_length_distribution():
    """
    Plot distribution of message lengths for spam vs ham
    """
    print("Generating message length distribution plot...")

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['length'] = df['message'].str.len()
    except:
        print("✗ Could not load dataset. Skipping length distribution plot.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    spam_lengths = df[df['label'] == 'spam']['length']
    ham_lengths = df[df['label'] == 'ham']['length']

    ax.hist(ham_lengths, bins=30, alpha=0.7, label='Ham', color='#27ae60', edgecolor='black')
    ax.hist(spam_lengths, bins=30, alpha=0.7, label='Spam', color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Message Length (characters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Message Lengths', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('message_length_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: message_length_distribution.png")
    plt.show()


def plot_cross_validation_scores():
    """
    Plot cross-validation scores
    """
    print("Generating cross-validation scores plot...")

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    except:
        print("✗ Could not load dataset. Skipping CV scores plot.")
        return

    # Train classifier with cross-validation
    classifier = SpamClassifier()
    X, y = classifier.preprocess_data(df)
    X_train, _, y_train, _ = classifier.split_data(X, y)
    scores = classifier.cross_validate(X_train, y_train, cv=5)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    folds = range(1, len(scores) + 1)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    ax.bar(folds, scores, color='#667eea', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')

    # Add error bands
    ax.fill_between(folds, mean_score - std_score, mean_score + std_score,
                     alpha=0.2, color='red', label=f'±1 Std: {std_score:.4f}')

    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_ylim([0.9, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('cv_scores.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cv_scores.png")
    plt.show()


def plot_prediction_confidence():
    """
    Plot prediction confidence distribution
    """
    print("Generating prediction confidence plot...")

    try:
        df = pd.read_csv('spam.csv', sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    except:
        print("✗ Could not load dataset. Skipping confidence plot.")
        return

    # Train classifier
    classifier = SpamClassifier()
    X, y = classifier.preprocess_data(df)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    classifier.train(X_train, y_train)

    # Get predictions and probabilities
    X_test_vec = classifier.vectorizer.transform(X_test)
    y_proba = classifier.classifier.predict_proba(X_test_vec)
    confidences = np.max(y_proba, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(confidences, bins=30, color='#667eea', edgecolor='black', linewidth=1.5)
    ax.axvline(x=np.mean(confidences), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(confidences):.4f}')

    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: prediction_confidence.png")
    plt.show()


def generate_all_visualizations():
    """
    Generate all visualizations
    """
    print("\n" + "="*70)
    print("SPAM CLASSIFIER - VISUALIZATION GENERATION")
    print("="*70 + "\n")

    visualizations = [
        ("1", "Confusion Matrix", plot_confusion_matrix),
        ("2", "Performance Metrics", plot_performance_metrics),
        ("3", "Class Distribution", plot_class_distribution),
        ("4", "ROC Curve", plot_roc_curve),
        ("5", "Message Length Distribution", plot_message_length_distribution),
        ("6", "Cross-Validation Scores", plot_cross_validation_scores),
        ("7", "Prediction Confidence", plot_prediction_confidence),
        ("8", "Generate All", None)
    ]

    print("Available visualizations:")
    for num, name, _ in visualizations:
        print(f"  {num}. {name}")

    choice = input("\nSelect visualization (1-8): ").strip()

    if choice == "8":
        print("\nGenerating all visualizations...\n")
        for num, name, func in visualizations[:-1]:
            try:
                func()
                print()
            except Exception as e:
                print(f"✗ Error generating {name}: {e}\n")
    else:
        for num, name, func in visualizations:
            if num == choice:
                try:
                    func()
                except Exception as e:
                    print(f"✗ Error: {e}")
                return

        print("Invalid choice. Please enter 1-8.")

    print("\n" + "="*70)
    print("All visualization images saved to current directory!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        generate_all_visualizations()
    except KeyboardInterrupt:
        print("\n✗ Cancelled by user.")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
