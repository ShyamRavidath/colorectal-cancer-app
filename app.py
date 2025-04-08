import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Load the .h5 model


# Load the .keras model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')
])
model.load_weights('Model_20_NWSE.h5')  # Load the .keras format

# Define classes
benign_classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM']
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
st.file_uploader(
    "Upload one or more images...",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the image
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Preprocess and predict
        img = preprocess_image(image)
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_idx]
        confidence_score = float(predictions[0][predicted_class_idx])
        result = "Benign" if predicted_class in benign_classes else "Malignant"

        # Show results
        st.markdown(f"**Tissue:** {result}")
        st.markdown(f"**Predicted Class:** {predicted_class}")
        st.markdown(f"**Confidence Score:** {confidence_score:.4f}")
        st.markdown("---")
