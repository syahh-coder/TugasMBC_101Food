# app_food101_hf.py
# --------------------------
# Streamlit Food-101 Image Classifier (Load model dari Hugging Face)
# --------------------------

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.express as px
from PIL import Image
import random
from huggingface_hub import hf_hub_download

# -----------------------------
# Page & Theme
# -----------------------------
st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="🍔🍣🍕",
    layout="wide",
)

st.title("🍔🍣 Food-101 Image Classifier")
st.markdown("Upload gambar makanan, model akan mengklasifikasikan salah satu dari **101 kelas Food-101**.")

# -----------------------------
# Hugging Face Model Loader
# -----------------------------
HF_REPO_ID = "Syahhh01/food101"  
MODEL_FILENAME = "model_food_101.h5" 
PROB_DECIMALS = 3

@st.cache_resource(show_spinner=True)
def load_my_model_from_hf(repo_id, filename):
    local_model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    model = load_model(local_model_path)
    return model

with st.spinner("Loading model from Hugging Face..."):
    try:
        model = load_my_model_from_hf(HF_REPO_ID, MODEL_FILENAME)
        IMG_SIZE = model.input_shape[1:3]  # ambil H,W dari model
        NUM_CLASSES = model.output_shape[-1]
        st.success(f"Model loaded. Input size: {IMG_SIZE}, Number of classes: {NUM_CLASSES}")
    except Exception as e:
        st.error("Gagal memuat model dari Hugging Face.")
        st.exception(e)
        st.stop()

# -----------------------------
# Food-101 labels langsung di script
# -----------------------------
LABEL_MAP = {
    0: "apple_pie", 1: "baby_back_ribs", 2: "baklava", 3: "beef_carpaccio", 4: "beef_tartare",
    5: "beet_salad", 6: "beignets", 7: "bibimbap", 8: "bread_pudding", 9: "breakfast_burrito",
    10: "bruschetta", 11: "caesar_salad", 12: "cannoli", 13: "caprese_salad", 14: "carrot_cake",
    15: "ceviche", 16: "cheesecake", 17: "cheese_plate", 18: "chicken_curry", 19: "chicken_quesadilla",
    20: "chicken_wings", 21: "chocolate_cake", 22: "chocolate_mousse", 23: "churros", 24: "clam_chowder",
    25: "club_sandwich", 26: "crab_cakes", 27: "creme_brulee", 28: "croque_madame", 29: "cup_cakes",
    30: "deviled_eggs", 31: "donuts", 32: "dumplings", 33: "edamame", 34: "eggs_benedict",
    35: "escargots", 36: "falafel", 37: "filet_mignon", 38: "fish_and_chips", 39: "foie_gras",
    40: "french_fries", 41: "french_onion_soup", 42: "french_toast", 43: "fried_calamari", 44: "fried_rice",
    45: "frozen_yogurt", 46: "garlic_bread", 47: "gnocchi", 48: "greek_salad", 49: "grilled_cheese_sandwich",
    50: "grilled_salmon", 51: "guacamole", 52: "gyoza", 53: "hamburger", 54: "hot_and_sour_soup",
    55: "hot_dog", 56: "huevos_rancheros", 57: "hummus", 58: "ice_cream", 59: "lasagna",
    60: "lobster_bisque", 61: "lobster_roll_sandwich", 62: "macaroni_and_cheese", 63: "macarons", 64: "miso_soup",
    65: "mussels", 66: "nachos", 67: "omelette", 68: "onion_rings", 69: "oysters",
    70: "pad_thai", 71: "paella", 72: "pancakes", 73: "panna_cotta", 74: "peking_duck",
    75: "pho", 76: "pizza", 77: "pork_chop", 78: "poutine", 79: "prime_rib",
    80: "pulled_pork_sandwich", 81: "ramen", 82: "ravioli", 83: "red_velvet_cake", 84: "risotto",
    85: "samosa", 86: "sashimi", 87: "scallops", 88: "seaweed_salad", 89: "shrimp_and_grits",
    90: "spaghetti_bolognese", 91: "spaghetti_carbonara", 92: "spring_rolls", 93: "steak", 94: "strawberry_shortcake",
    95: "sushi", 96: "tacos", 97: "takoyaki", 98: "tiramisu", 99: "tuna_tartare",
    100: "waffles"
}

# Warna random untuk badge
random.seed(42)
CLASS_COLORS = {label: f"#{random.randint(0,0xFFFFFF):06x}" for label in LABEL_MAP.values()}

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload gambar makanan (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = img.resize(IMG_SIZE)
    x = image.img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1,H,W,3)

    # Prediksi
    probs = model.predict(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_idx]
    top_prob = float(probs[pred_idx])

    # Badge
    badge_color = CLASS_COLORS.get(pred_label, "#1f77b4")
    st.markdown(
        f"<div style='padding:10px;border-radius:12px;background:{badge_color};color:white;display:inline-block;'>"
        f"Prediksi: <b>{pred_label}</b> • Prob: {top_prob:.{PROB_DECIMALS}f}"
        f"</div>", unsafe_allow_html=True
    )

    # Bar chart probabilitas top 10
    top_indices = probs.argsort()[-10:][::-1]
    df_probs = {
        "Class": [LABEL_MAP[i] for i in top_indices],
        "Probability": [probs[i] for i in top_indices]
    }
    fig = px.bar(
        df_probs, x="Class", y="Probability",
        text=[f"{p:.{PROB_DECIMALS}f}" for p in df_probs["Probability"]],
        title="Top 10 Class Probabilities", range_y=[0, 1]
    )
    fig.update_traces(
        marker_color=[CLASS_COLORS.get(c, None) for c in df_probs["Class"]],
        textposition="outside"
    )
    st.plotly_chart(fig, use_container_width=True)
