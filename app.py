import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Model.h5')

# Define classes
CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Colorectal Cancer Classifier")
st.write("Upload a histopathology image to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make a prediction
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASSES[predicted_class_idx]
    confidence_score = float(predictions[0][predicted_class_idx])

    # Display the result
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")
