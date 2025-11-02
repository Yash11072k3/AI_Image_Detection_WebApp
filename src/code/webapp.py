import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from utils import load_model, weighted_voting
from model_definitions import CNN, ResNetModel, PatchSelection, DIF, UFD
import torch.nn.functional as F
import numpy as np
import hashlib
import sqlite3
import re

# Set page config
st.set_page_config(
    page_title="AI Image Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====== Main App CSS ======
custom_css = """
<style>
    }
    .stFileUploader {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid #00c6ff !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    .stFileUploader label {
        color: white !important;
        font-size: 18px !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stFileUploader > div > div {
        background-color: rgba(40, 40, 40, 0.7) !important;
        border: 2px dashed #00c6ff !important;
        border-radius: 8px !important;
    }
    .stFileUploader > div > div > button {
        background-color: #00c6ff !important;
        color: black !important;
        font-weight: bold !important;
        border: none !important;
    }
    h1 {
        color: #00ffcc !important;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 30px;
    }
    .uploaded-img {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .uploaded-img > img {
        border: 3px solid #00c6ff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        max-width: 100%;
        max-height: 400px;
        object-fit: contain;
    }
    .prediction-container {
        background: linear-gradient(135deg, #1a2a3a 0%, #0e1117 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 30px auto;
        border: 2px solid #00ffcc;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        max-width: 600px;
    }
    .prediction-header {
        font-size: 24px;
        font-weight: bold;
        color: #00ffcc;
        margin-bottom: 15px;
        text-align: center;
    }
    .prediction-result {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        border-radius: 10px;
    }
    .real {
        color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        border: 2px solid #4CAF50;
    }
    .fake {
        color: #F44336;
        background-color: rgba(244, 67, 54, 0.1);
        border: 2px solid #F44336;
    }
    .confidence-meter {
        height: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin: 20px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .confidence-real {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    .confidence-fake {
        background: linear-gradient(90deg, #F44336, #FF9800);
    }
    .model-details {
        margin-top: 25px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 15px;
    }
    .model-row {
        display: flex;
        justify-content: space-between;
        margin: 8px 0;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
    }
    .model-name {
        font-weight: bold;
        color: #00c6ff;
    }
    .model-prediction {
        font-weight: bold;
    }
    .model-real {
        color: #8BC34A;
    }
    .model-fake {
        color: #FF9800;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;          
        right: 0;         
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #666;
        font-size: 12px;
        background: transparent; 
        z-index: 9999;           
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ====== Model Loading ======
MODEL_PATH = "D:/AI_Image_Detection_WebApp/src/newly_trained_model"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["fake", "real"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    models = [
        load_model(CNN, "CNN", MODEL_PATH, DEVICE),
        load_model(ResNetModel, "ResNetModel", MODEL_PATH, DEVICE),
        load_model(PatchSelection, "PatchSelection", MODEL_PATH, DEVICE),
        load_model(DIF, "DIF", MODEL_PATH, DEVICE),
        load_model(UFD, "UFD", MODEL_PATH, DEVICE),
    ]
    return models

models = load_models()
weights = [1.0] * len(models)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ====== Prediction Functions ======
def get_model_predictions(image_tensor):
    preds = []
    raw_outputs = []
    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            preds.append(pred.item())
            raw_outputs.append(output.cpu().numpy()[0])
    return preds, raw_outputs

def calculate_confidence(raw_outputs):
    avg_probs = np.mean([F.softmax(torch.from_numpy(output), dim=0).numpy() for output in raw_outputs], axis=0)
    confidence = np.max(avg_probs) * 100
    return confidence, avg_probs

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    preds, raw_outputs = get_model_predictions(image_tensor)
    final_prediction = weighted_voting(preds, weights, len(CLASS_NAMES))
    confidence, class_probs = calculate_confidence(raw_outputs)
    
    return {
        "final_prediction": CLASS_NAMES[final_prediction],
        "confidence": confidence,
        "class_probs": class_probs,
        "model_details": [
            {
                "name": model.__class__.__name__,
                "prediction": CLASS_NAMES[pred],
                "raw_output": raw_output
            }
            for model, pred, raw_output in zip(models, preds, raw_outputs)
        ]
    }
def display_prediction_result(prediction_data):
    is_real = prediction_data["final_prediction"] == "real"
    result_color = "#4CAF50" if is_real else "#F44336"
    result_text = "REAL IMAGE" if is_real else "AI GENERATED"
    
    st.markdown(f"""
    <div class="prediction-container">
        <div style="text-align: center; font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;">
            Analysis Result
        </div>
        <div class="prediction-result" style="color: {result_color}; border-color: {result_color}">
            {result_text}
        </div>
        <div class="confidence-meter">
            <div class="confidence-fill {'confidence-real' if is_real else 'confidence-fake'}" 
                 style="width: {prediction_data['confidence']}%"></div>
        </div>
        <div style="text-align: center; margin: 1rem 0;">
            Confidence: {prediction_data["confidence"]:.1f}%
        </div>
        <div style="margin: 1rem 0;">
            <p style="text-align: center; font-weight: bold;">Probability Breakdown:</p>
            <div style="display: flex; justify-content: center; gap: 2rem;">
                <div>Real: {prediction_data["class_probs"][1]*100:.1f}%</div>
                <div>AI Generated: {prediction_data["class_probs"][0]*100:.1f}%</div>
            </div>
        </div>
        <div style="margin-top: 1.5rem;">
            <p style="text-align: center; font-weight: bold;">Model Predictions:</p>
    """, unsafe_allow_html=True)
    
    for model in prediction_data["model_details"]:
        pred_color = "#8BC34A" if model["prediction"] == "real" else "#FF9800"
        st.markdown(f"""
        <div class="model-row">
            <span class="model-name">{model["name"]}</span>
            <span class="model-prediction" style="color: {pred_color}">
                {model["prediction"].upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ====== Main App ======
def main_app():
    # Initialize session state variable
    if "username" not in st.session_state:
        st.session_state.username = "Folks"  # or set it to a value from login input
  
    st.markdown("<h1> AI Image Detection</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; margin-bottom: 20px;'>Welcome, {st.session_state.username}!</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 30px;
        border-left: 4px solid #00ffcc;
    ">
    <p style="font-size: 16px;text-align: center; line-height: 1.6;">
    This image detection system analyzes images using an ensemble of deep learning models to identify the image whether it is real or AI-generated.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Upload an image to analyze", 
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, PNG formats up to 20MB"
    )
    
    if uploaded_image is not None:
        try:
            with st.spinner("Analyzing image..."):
                image = Image.open(uploaded_image).convert("RGB")
                st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                prediction_data = predict_image(image)
                display_prediction_result(prediction_data)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    st.markdown('<div class="footer">AI Image Detection System v1.0| 2025 </div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main_app()