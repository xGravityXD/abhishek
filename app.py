import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model('scene_classifier.h5')
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("Natural Scene Classifier (Batch Prediction)")

uploaded_files = st.file_uploader(
    "Upload one or more images of natural scenes...", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded.**")
    for uploaded_file in uploaded_files:
        cols = st.columns(2)
        # Display the image
        img = Image.open(uploaded_file).convert('RGB')
        cols[0].image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x)
        pred_index = np.argmax(preds, axis=1)[0]
        pred_label = class_labels[pred_index]
        confidence = preds[0][pred_index]

        # Display prediction
        cols[1].markdown(f"### Prediction: **{pred_label.capitalize()}**")
        cols[1].markdown(f"**Confidence:** {confidence:.2%}")