import streamlit as st
from transformers import pipeline
import os
import zipfile
import re
import google.generativeai as genai

# --- Auto-Unzip Logic for Deployment ---
# This is the folder name that is INSIDE the zip file
model_folder = 'climatebert_model' 
# This is the zip file we uploaded to GitHub
zip_archive = 'climatebert_model_archive.zip' 

# This runs ONCE when the app starts in the cloud
if not os.path.exists(model_folder) and os.path.exists(zip_archive):
    print(f"'{model_folder}' not found. Unzipping '{zip_archive}'...")
    with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
        zip_ref.extractall('.') # Unzip to the current directory
    print("✅ Model unzipped successfully!")
# ---

# --- Model Loading ---
# We cache the model so it only loads once
@st.cache_resource
def load_classifier():
    """Loads the fine-tuned ClimateBERT pipeline from the unzipped folder."""
    
    # Check if the unzipped folder exists
    if not os.path.exists(model_folder):
        st.error(f"Error: Model folder '{model_folder}' not found.")
        st.error("This can happen during the first boot. The app will restart in a moment.")
        st.error("If this persists, please check 'climatebert_model_archive.zip' is in the GitHub repo.")
        return None
        
    try:
        # Load the model from the folder
        classifier = pipeline(
            "text-classification",
            model=model_folder,
            tokenizer=model_folder
        )
        print("✅ Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading model from folder: {e}")
        return None

# --- Page Configuration (Set up the browser tab) ---
st.set_page_config(
    page_title="Climate Intelligence Assistant",
    page_icon="🌍",
    layout="wide"
)

# --- Sidebar for API Key ---
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key",
    type="password",
    help="Get your free key from [Google AI Studio](https://aistudio.google.com/app/apikey)"
)

# --- Main Page UI ---
st.title("🌍 Climate Intelligence Assistant")
st.write("An AI-powered tool to analyze climate-related text for source credibility and provide generative AI explanations.")

# Load the model
classifier = load_classifier()

# If the model fails to load, stop the app.
if classifier is None:
    st.stop()

# --- Input Fields ---
article_title = st.text_input("Article Title", placeholder="Enter article title here...")
article_text = st.text_area("Article Text", placeholder="Paste article text here...", height=250)

# --- Analyze Button ---
if st.button("Analyze Content", use_container_width=True, type="primary"):
    
    # --- Input Validation ---
    if not article_title and not article_text:
        st.warning("Please enter a title or some text to analyze.")
    elif not gemini_api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to get AI explanations.")
    else:
        # Show a loading spinner while processing
        with st.spinner("Analyzing... This may take a moment."):
            try:
                # --- 1. Run ClimateBERT Classification ---
                combined_text = f"{article_title} {re.sub(re.compile('<.*?>'), '', article_text)}"
                prediction = classifier(combined_text)[0]
                label = prediction['label']
                score = prediction['score']

                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                # Display the results in columns
                with col1:
                    st.metric("Predicted Source", label.upper())
                with col2:
                    st.metric("Confidence Score", f"{score:.2%}")

                # --- 2. Configure and Call Gemini API ---
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
                
                # Create a clear prompt for the LLM
                prompt = f"""
                An AI model has classified the following article as '{label}' with {score:.0%} confidence.
                Please provide a brief, easy-to-understand explanation (in Hinglish, as if explaining to a friend) about what this classification means.
                Also, point out any potential red flags in the text itself, based on the classification.

                Article Title: {article_title}
                Article Text: {article_text}
                """
                
                response = model.generate_content(prompt)
                
                st.subheader("🤖 AI-Generated Explanation (from Gemini)")
                st.info(response.text) # Display Gemini's response

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Please check your API key and try again.")