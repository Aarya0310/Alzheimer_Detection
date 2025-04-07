import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

# Load the model safely
@st.cache_resource
def load_vgg19_model():
    try:
        model_path = "vgg19.h5"
        if not os.path.exists(model_path):
            st.error("❌ Model file not found: vgg19.h5")
            return None
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Load training history from .pkl
@st.cache_data
def load_training_history():
    try:
        with open("/mnt/data/vgg19_history.pkl", "rb") as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        st.warning(f"⚠️ Could not load training history: {e}")
        return None

# Show spinner while loading
with st.spinner("Loading model..."):
    model = load_vgg19_model()

if model is None:
    st.stop()  # Stop app if model failed to load

# Load training history
history = load_training_history()

# Define class labels
class_names = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# Title
st.title("🧠 Alzheimer's Detection from Brain Scans")
st.write("Upload an MRI brain scan and the model will predict the likelihood of Alzheimer's.")

# Show training history chart
if history:
    st.subheader("📈 Model Training History")
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
    ax.set_title("Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict and apply softmax
    prediction = model.predict(img_array)
    probabilities = tf.nn.softmax(prediction[0]).numpy()
    predicted_class = class_names[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100

    # Show results
    st.subheader("🧪 Prediction")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Visualize probabilities
    st.bar_chart(probabilities)
