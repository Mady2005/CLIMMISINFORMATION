import streamlit as st
from transformers import pipeline
import os
import zipfile
import re
import google.generativeai as genai

# --- Auto-Unzip Logic ---
model_folder = 'climatebert_model' 
zip_archive = 'climatebert_model_archive.zip' 

if not os.path.exists(model_folder) and os.path.exists(zip_archive):
    print(f"'{model_folder}' not found. Unzipping '{zip_archive}'...")
    with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("✅ Model unzipped successfully!")

# --- Model Loading ---
@st.cache_resource
def load_classifier():
    if not os.path.exists(model_folder):
        st.error(f"Error: Model folder '{model_folder}' not found.")
        return None
    try:
        classifier = pipeline("text-classification", model=model_folder, tokenizer=model_folder)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Page Config ---
st.set_page_config(page_title="Climate Intelligence Assistant", page_icon="🌍", layout="wide")

# --- Sidebar ---
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

# --- Main UI ---
st.title("🌍 Climate Intelligence Assistant")
st.write("An AI-powered tool to analyze climate-related text.")

classifier = load_classifier()
if classifier is None: st.stop()

article_title = st.text_input("Article Title")
article_text = st.text_area("Article Text", height=250)

if st.button("Analyze Content", type="primary"):
    if not article_title and not article_text:
        st.warning("Please enter text.")
    elif not gemini_api_key:
        st.warning("Please enter API Key.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # 1. Classification
                combined_text = f"{article_title} {re.sub(r'<.*?>', '', article_text)}"
                prediction = classifier(combined_text)[0]
                label = prediction['label']
                score = prediction['score']

                col1, col2 = st.columns(2)
                col1.metric("Predicted Source", label.upper())
                col2.metric("Confidence", f"{score:.2%}")

                # 2. Gemini Call (Using YOUR specific models)
                genai.configure(api_key=gemini_api_key)
                
                model = None
                # These are the exact models from your list
                model_options = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest']
                
                for name in model_options:
                    try:
                        test_model = genai.GenerativeModel(name)
                        test_model.generate_content("Hello") 
                        model = test_model
                        print(f"✅ Connected to {name}")
                        break 
                    except:
                        continue
                
                if model is None:
                    st.error("Could not connect to Gemini. Please check your API Key.")
                else:
                    prompt = f"""
                    The article is classified as '{label}' ({score:.0%} confidence).
                    Explain why in simple Hinglish. Point out red flags.
                    Title: {article_title}
                    Text: {article_text}
                    """
                    response = model.generate_content(prompt)
                    st.subheader("🤖 AI Explanation")
                    st.info(response.text)

            except Exception as e:
                st.error(f"An error occurred: {e}")