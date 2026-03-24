"""
Flask Web Interface for Spam Message Classifier
Provides a user-friendly web interface for spam detection
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
from datetime import datetime
import json
import csv
from io import StringIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Defaults are set first so routes can still run even if imports fail on serverless.
classifier = None
model_loaded = False
db_ready = False
db_functions_available = False
startup_errors = []

# Placeholders so names always exist for route handlers and tests.
init_db = None
save_prediction = None
get_prediction_logs = None
count_prediction_logs = None
get_prediction_logs_for_export = None

# Import database helpers lazily and safely.
try:
    from db import (
        init_db,
        save_prediction,
        get_prediction_logs,
        count_prediction_logs,
        get_prediction_logs_for_export,
    )
    db_functions_available = True
except Exception as e:
    startup_errors.append(f"DB import error: {e}")
    print(f"✗ Warning: Could not import database helpers: {e}")

# Import ML classifier lazily and safely.
try:
    from spam_classifier import SpamClassifier
    classifier = SpamClassifier()
except Exception as e:
    startup_errors.append(f"ML import error: {e}")
    print(f"✗ Warning: Could not import spam classifier: {e}")


def _train_with_seed_data():
    """Train a minimal fallback model when dataset/artifacts are unavailable."""
    global model_loaded

    if classifier is None:
        return False

    ham_samples = [
        "Hey, are we still meeting today?",
        "Can you call me when you arrive?",
        "I will be late to class today.",
        "Do not forget to bring your assignment.",
        "Are you coming for the game tonight?",
        "Thanks for your purchase. Your order is confirmed.",
        "Please review the attached report before 5pm.",
        "Lunch at 1pm works for me.",
        "I sent the document to your email.",
        "Let us reschedule the meeting to tomorrow.",
    ]

    spam_samples = [
        "Congratulations! You have won $1000. Click here to claim now!",
        "URGENT! Your account has been compromised. Send your details now!",
        "Free entry in a weekly competition! Text WIN to 12345.",
        "You have been selected for a prize. Call now!",
        "Earn money fast from home with this simple trick.",
        "Limited offer! Claim your reward immediately.",
        "Winner! You have been chosen for a cash bonus.",
        "Act now to verify your account and avoid suspension.",
        "Click this link to receive your free voucher.",
        "You are pre-approved for instant cash. Apply now.",
    ]

    X_train = ham_samples + spam_samples
    y_train = [0] * len(ham_samples) + [1] * len(spam_samples)

    classifier.train(X_train, y_train)
    classifier.save_model()
    model_loaded = True
    return True


def ensure_model_loaded():
    """Guarantee the model is ready: load existing artifacts or train a fallback."""
    global model_loaded

    if classifier is None:
        return False

    if model_loaded:
        return True

    # 1) Try loading pre-trained artifacts.
    try:
        classifier.load_model()
        model_loaded = True
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        startup_errors.append(f"Model load error: {e}")
        print(f"✗ Warning: Could not load model artifacts: {e}")

    # 2) Try training from dataset if available in runtime filesystem.
    try:
        dataset_path = 'spam.csv'
        if os.path.exists(dataset_path):
            print("ℹ Training model from spam.csv at startup...")
            df = classifier.load_data(dataset_path)
            if df is not None:
                X, y = classifier.preprocess_data(df)
                classifier.train(X, y)
                classifier.save_model()
                model_loaded = True
                print("✓ Model trained from dataset and ready.")
                return True
    except Exception as e:
        startup_errors.append(f"Dataset bootstrap error: {e}")
        print(f"✗ Warning: Could not train from dataset: {e}")

    # 3) Final fallback: train a small in-memory seed model.
    try:
        print("ℹ Training fallback seed model at startup...")
        if _train_with_seed_data():
            print("✓ Fallback seed model trained and ready.")
            return True
    except Exception as e:
        startup_errors.append(f"Seed bootstrap error: {e}")
        print(f"✗ Warning: Could not train fallback seed model: {e}")

    return False

# Try to initialize database on startup
if db_functions_available and init_db is not None:
    try:
        init_db()
        db_ready = True
        print("✓ Database connected and table is ready!")
    except Exception as e:
        startup_errors.append(f"DB init error: {e}")
        print(f"✗ Warning: Could not initialize database: {e}")
        print("  Prediction logging is disabled until DATABASE_URL is configured correctly.")

# Ensure model is available on startup.
ensure_model_loaded()


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/favicon.ico')
def favicon():
    """Return an empty favicon response to avoid browser 404 noise."""
    return Response(status=204)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for single message prediction"""
    if not model_loaded and not ensure_model_loaded():
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 503

    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON payload', 'success': False}), 400

        message = str(data.get('message', '')).strip()

        if not message:
            return jsonify({'error': 'Message cannot be empty', 'success': False}), 400

        if len(message) > 500:
            return jsonify({'error': 'Message too long (max 500 characters)', 'success': False}), 400

        result = classifier.predict(message)
        if not isinstance(result, dict):
            return jsonify({'error': 'Prediction model unavailable', 'success': False}), 503

        if 'spam_probability' not in result or 'ham_probability' not in result or 'label' not in result:
            return jsonify({'error': 'Invalid prediction output', 'success': False}), 500

        # Determine risk level
        spam_prob = result['spam_probability']
        if spam_prob > 0.7:
            risk_level = 'HIGH'
        elif spam_prob > 0.3:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'

        db_log_saved = False
        db_log_id = None

        # Save prediction to PostgreSQL if available.
        if db_ready:
            try:
                db_log_id = save_prediction(
                    message=message,
                    predicted_label=result['label'],
                    spam_probability=result['spam_probability'],
                    ham_probability=result['ham_probability'],
                    risk_level=risk_level,
                )
                db_log_saved = True
            except Exception as db_error:
                print(f"✗ Warning: Failed to save prediction log: {db_error}")

        return jsonify({
            'success': True,
            'classification': result['label'],
            'spam_probability': round(result['spam_probability'], 4),
            'ham_probability': round(result['ham_probability'], 4),
            'risk_level': risk_level,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'db_log_saved': db_log_saved,
            'db_log_id': db_log_id
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """API endpoint for batch predictions"""
    if not model_loaded and not ensure_model_loaded():
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 503

    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON payload', 'success': False}), 400

        messages = data.get('messages', [])
        if not isinstance(messages, list):
            return jsonify({'error': 'messages must be an array', 'success': False}), 400

        messages = [str(msg).strip() for msg in messages if str(msg).strip()]

        if not messages:
            return jsonify({'error': 'No messages provided', 'success': False}), 400

        if len(messages) > 100:
            return jsonify({'error': 'Too many messages (max 100)', 'success': False}), 400

        results = classifier.predict_batch(messages)
        if not isinstance(results, list):
            return jsonify({'error': 'Prediction model unavailable', 'success': False}), 503

        predictions = []
        spam_count = 0
        ham_count = 0

        for msg, result in zip(messages, results):
            if not isinstance(result, dict):
                continue

            if 'label' not in result or 'spam_probability' not in result or 'ham_probability' not in result:
                continue

            if result['label'] == 'SPAM':
                spam_count += 1
            else:
                ham_count += 1

            predictions.append({
                'message': msg,
                'classification': result['label'],
                'spam_probability': round(result['spam_probability'], 4),
                'ham_probability': round(result['ham_probability'], 4)
            })

        if not predictions:
            return jsonify({'error': 'No valid predictions produced', 'success': False}), 500

        return jsonify({
            'success': True,
            'predictions': predictions,
            'summary': {
                'total': len(messages),
                'spam': spam_count,
                'ham': ham_count,
                'spam_percentage': round(spam_count / len(messages) * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    max_features = getattr(classifier.vectorizer, 'max_features', 12000) if classifier else 12000
    return jsonify({
        'model_loaded': model_loaded,
        'model_type': 'Multinomial Naive Bayes',
        'vectorizer_type': 'TF-IDF',
        'max_features': max_features,
        'expected_accuracy': '98.5%',
        'expected_precision': '98.9%',
        'expected_recall': '95.4%'
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent prediction logs with pagination and optional search."""
    if not db_ready:
        return jsonify({
            'success': False,
            'error': 'Database logging is not available. Configure DATABASE_URL first.'
        }), 503

    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
    except ValueError:
        return jsonify({'success': False, 'error': 'page and page_size must be integers'}), 400

    search = request.args.get('q', '').strip()
    sort = request.args.get('sort', 'newest').strip()

    if page < 1:
        return jsonify({'success': False, 'error': 'page must be >= 1'}), 400

    if page_size < 1 or page_size > 100:
        return jsonify({'success': False, 'error': 'page_size must be between 1 and 100'}), 400

    try:
        logs = get_prediction_logs(page=page, page_size=page_size, search=search, sort=sort)
        total = count_prediction_logs(search=search)
        total_pages = (total + page_size - 1) // page_size

        return jsonify({
            'success': True,
            'logs': logs,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': total_pages
            },
            'search': search,
            'sort': sort
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/export', methods=['GET'])
def export_logs_csv():
    """Export filtered logs as CSV."""
    if not db_ready:
        return jsonify({
            'success': False,
            'error': 'Database logging is not available. Configure DATABASE_URL first.'
        }), 503

    search = request.args.get('q', '').strip()
    sort = request.args.get('sort', 'newest').strip()
    try:
        max_rows = int(request.args.get('max_rows', 10000))
    except ValueError:
        return jsonify({'success': False, 'error': 'max_rows must be an integer'}), 400

    try:
        logs = get_prediction_logs_for_export(search=search, sort=sort, max_rows=max_rows)

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'id',
            'message',
            'predicted_label',
            'spam_probability',
            'ham_probability',
            'risk_level',
            'created_at'
        ])

        for log in logs:
            writer.writerow([
                log.get('id'),
                log.get('message', ''),
                log.get('predicted_label', ''),
                log.get('spam_probability', ''),
                log.get('ham_probability', ''),
                log.get('risk_level', ''),
                log.get('created_at', ''),
            ])

        csv_data = output.getvalue()
        output.close()

        filename = f"prediction_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    return jsonify({
        'app_name': 'Spam Message Identifier',
        'version': '1.0.0',
        'model_loaded': model_loaded,
        'database_logging_enabled': db_ready,
        'supported_languages': ['English'],
        'max_message_length': 500,
        'max_batch_size': 100
    })


@app.route('/api/health', methods=['GET'])
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'db_ready': db_ready,
        'startup_errors': startup_errors,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found', 'success': False}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'success': False}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPAM MESSAGE IDENTIFIER - FLASK WEB INTERFACE")
    print("="*70)
    print("\n🚀 Starting Flask server...")
    print("\n📍 Open your browser and go to: http://localhost:5000")
    print("\n📚 API Endpoints:")
    print("   POST /api/predict          - Classify a single message")
    print("   POST /api/predict-batch    - Classify multiple messages")
    print("   GET  /api/logs             - View prediction logs (pagination/search)")
    print("   GET  /api/logs/export      - Export prediction logs as CSV")
    print("   GET  /api/model-info       - Get model information")
    print("   GET  /api/stats            - Get application statistics")
    print("   GET  /api/health           - Health check")
    print("\n" + "="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
