import os
import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import threading
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path('models/Experiment2/baseline_cnn.keras')
model = None

class_indices = np.load(resource_path('data/preprocessed/class_indices.npy'), allow_pickle=True).item()
index_to_class = {v: k for k, v in class_indices.items()}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def load_model_if_needed():
    global model
    if model is None:
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    return True

@app.route('/predict', methods=['POST'])
def predict():
    if not load_model_if_needed():
        return jsonify({
            'success': False,
            'error': 'Failed to load model'
        }), 500

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file part'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            processed_img = load_and_preprocess_image(file_path)

            predictions = model.predict(processed_img)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx] * 100)
            pred_class = index_to_class[pred_idx]

            return jsonify({
                'success': True,
                'prediction': {
                    'disease': pred_class,
                    'confidence': confidence
                }
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return jsonify({
        'success': False,
        'error': 'Invalid file format. Please upload an image (png, jpg, jpeg).'
    }), 400

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_vue(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    load_model_if_needed()
    threading.Thread(target=open_browser).start()
    app.run(debug=False, host='0.0.0.0', port=5000)