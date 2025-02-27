from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the model
try:
    model = load_model('Model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define classes
CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the uploaded file
    uploads_dir = os.path.join('static', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)  # Create the uploads directory if it doesn't exist
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Preprocess the image and make a prediction
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASSES[predicted_class_idx]
    confidence_score = float(predictions[0][predicted_class_idx])

    # Return the result
    return jsonify({
        'predicted_class': predicted_class,
        'confidence_score': confidence_score,
        'image_url': file_path
    })

if __name__ == '__main__':
    app.run(debug=True)
