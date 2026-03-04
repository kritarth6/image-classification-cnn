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

# Dark theme CSS
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
}

.subtitle{
text-align:center;
font-size:20px;
margin-bottom:30px;
color:#d3d3d3;
}

.prediction-card{
background:#1f2a40;
padding:25px;
border-radius:15px;
text-align:center;
box-shadow:0px 6px 20px rgba(0,0,0,0.5);
}

.prediction-text{
font-size:32px;
font-weight:bold;
color:#00ffd5;
}

.footer{
text-align:center;
margin-top:40px;
font-size:16px;
color:#cccccc;
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

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32,32))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = float(np.max(tf.nn.softmax(prediction)))

    with col2:

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        st.markdown("### Prediction")

        st.markdown(
            f"<div class='prediction-text'>{predicted_class}</div>",
            unsafe_allow_html=True
        )

        st.progress(confidence)

        st.write(f"Confidence: **{confidence:.2%}**")

        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.markdown(
    "<div class='footer'>Built with ❤️ by <b>Kritarth Joshi</b></div>",
    unsafe_allow_html=True
)
