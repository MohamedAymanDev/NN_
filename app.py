import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Arabic Sign Language AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(
        135deg,
        #0F172A,
        #111827,
        #0B1120
    );
    color: white;
}

/* TITLE */

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: 900;
    background: linear-gradient(to right, #00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #C7D2FE;
    margin-bottom: 40px;
}

/* CARD */

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 25px rgba(0,255,170,0.15);
}

/* RESULT */

.result-box {
    background: linear-gradient(
        145deg,
        rgba(0,255,170,0.12),
        rgba(0,217,245,0.08)
    );

    border: 1px solid rgba(255,255,255,0.1);

    padding: 35px;
    border-radius: 25px;

    text-align: center;

    box-shadow: 0 0 30px rgba(0,255,170,0.2);
}

/* PREDICTION */

.prediction {
    font-size: 50px;
    font-weight: bold;
    color: #00FFAA;
    margin-bottom: 15px;
}

/* CONFIDENCE */

.confidence {
    font-size: 24px;
    color: white;
}

/* IMAGE */

img {
    border-radius: 20px !important;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 15px;
}

/* FOOTER */

.footer {
    text-align:center;
    margin-top:40px;
    color:#94A3B8;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = tf.keras.models.load_model(
    "best_cnn_model.keras",
    compile=False
)

# ==========================================
# CLASS LABELS
# ==========================================
idx_to_class = {
    0:'ain',
    1:'al',
    2:'aleff',
    3:'bb',
    4:'dal',
    5:'dha',
    6:'dhad',
    7:'fa',
    8:'gaaf',
    9:'ghain',
    10:'ha',
    11:'haa',
    12:'jeem',
    13:'kaaf',
    14:'khaa',
    15:'la',
    16:'laam',
    17:'meem',
    18:'nun',
    19:'ra',
    20:'saad',
    21:'seen',
    22:'sheen',
    23:'ta',
    24:'taa',
    25:'thaa',
    26:'thal',
    27:'toot',
    28:'waw',
    29:'ya',
    30:'yaa',
    31:'zay'
}

# ==========================================
# HEADER
# ==========================================
st.markdown(
    """
    <div class="main-title">
        🧠 Arabic Sign Language AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Upload a hand sign image and let AI predict it instantly 🚀
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# MAIN APP
# ==========================================
if uploaded_file is not None:

    col1, col2 = st.columns([1,1])

    # ======================================
    # IMAGE SECTION
    # ======================================
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ======================================
    # PREPROCESSING
    # ======================================
    image = image.resize((64, 64))

    img_array = np.array(image) / 255.0

    # shape => (1,64,64,1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # ======================================
    # PREDICTION
    # ======================================
    with st.spinner("🤖 AI is analyzing image..."):

        time.sleep(1)

        prediction = model.predict(img_array)

    pred_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    predicted_label = idx_to_class[pred_class]

    # ======================================
    # RESULT SECTION
    # ======================================
    with col2:

        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.markdown(f"### 🔮 {predicted_label}")

        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        st.progress(confidence)

        st.subheader("📊 Top Predictions")

        top_3 = np.argsort(prediction[0])[-3:][::-1]

        for i in top_3:

            st.write(
                f"✅ {idx_to_class[i]} — {prediction[0][i]*100:.2f}%"
            )

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
    <div class="footer">
        🚀 Powered by CNN + TensorFlow + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)