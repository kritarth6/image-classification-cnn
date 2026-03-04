import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg, #1f4037, #99f2c8);
color: white;
}

.title {
text-align:center;
font-size:50px;
font-weight:bold;
}

.subtitle{
text-align:center;
font-size:20px;
margin-bottom:30px;
}

.result{
font-size:30px;
font-weight:bold;
color:#00ffcc;
}

.footer{
text-align:center;
margin-top:50px;
font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("cnn_model.h5")

class_names = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

# Header
st.markdown("<div class='title'>🧠 AI Image Classifier</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Deep Learning CNN Model | Upload Image to Predict Object</div>", unsafe_allow_html=True)

st.divider()

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1,col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32,32))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    with col2:

        st.markdown("### Prediction")

        st.markdown(
            f"<div class='result'>{predicted_class}</div>",
            unsafe_allow_html=True
        )

        st.progress(float(confidence))

        st.write(f"Confidence: **{confidence:.2%}**")

st.divider()

# Footer
st.markdown(
    "<div class='footer'>Built with ❤️ by <b>Kritarth Joshi</b></div>",
    unsafe_allow_html=True
)