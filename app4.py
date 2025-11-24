import streamlit as st
from transformers import pipeline
import os
import zipfile
import re
import google.generativeai as genai
from PIL import Image

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
st.set_page_config(
    page_title="Climate Intelligence Assistant",
    page_icon="🌍",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
st.sidebar.info("Get your free key from [Google AI Studio](https://aistudio.google.com/app/apikey)")

# --- Main Header ---
st.title("🌍 Climate Intelligence Assistant")
st.markdown("### 🛡️ The All-in-One Climate Misinformation Detective")
st.caption("Analyzes Text, Detects Logical Fallacies, and Debunks Misleading Images.")

# Load Model
classifier = load_classifier()
if classifier is None: st.stop()

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🕵️ Analyze Content (Text & Images)", "ℹ️ About the Project"])

# --- TAB 1: ANALYSIS (Main Tool) ---
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("1. Input Data")
        article_title = st.text_input("Headline / Title", placeholder="e.g., New study shows ice is growing...")
        article_text = st.text_area("Article Text", height=150, placeholder="Paste the content here...")
        
        st.markdown("---")
        st.write("📷 **Upload Evidence (Optional)**")
        uploaded_image = st.file_uploader("Upload a chart, meme, or photo to analyze", type=["jpg", "png", "jpeg"])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Evidence", use_column_width=True)

    with col_right:
        st.subheader("2. AI Analysis Report")
        analyze_btn = st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True)

        if analyze_btn:
            if not gemini_api_key:
                st.error("🔑 Please enter your Gemini API Key in the sidebar.")
            elif not article_title and not article_text and not uploaded_image:
                st.warning("⚠️ Please provide at least Text OR an Image to analyze.")
            else:
                with st.spinner("🤖 Processing... Detecting fallacies & analyzing visuals..."):
                    try:
                        # --- A. Text Classification (ClimateBERT) ---
                        # Only run BERT if there is text
                        label = "N/A"
                        score = 0.0
                        
                        if article_text or article_title:
                            combined_text = f"{article_title} {re.sub(r'<.*?>', '', article_text)}"
                            # Truncate for BERT if too long (simple fix)
                            prediction = classifier(combined_text[:512])[0] 
                            label = prediction['label']
                            score = prediction['score']

                        # --- B. Gemini Call (Multimodal) ---
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash') # 2.0 Flash supports images well
                        
                        # Construct inputs for Gemini
                        inputs = []
                        
                        # The Prompt
                        prompt_text = f"""
                        You are an expert Climate Misinformation Analyst. Analyze the following inputs.
                        
                        Context from BERT Classifier: Predicted Source = {label} ({score:.0%} Confidence).
                        
                        Please generate a report with these sections:
                        1. **Verdict:** (Credible, Misleading, or Mixed? Start with an Emoji).
                        2. **Logical Fallacy Check:** (Did the text use Strawman, Cherry Picking, or Ad Hominem? Explain).
                        3. **Visual Analysis:** (If an image is provided: Is the graph misleading? Is the photo out of context? If no image, skip this).
                        4. **Correction:** (Correct the misinformation with scientific facts).
                        
                        Headline: {article_title}
                        Text: {article_text}
                        """
                        inputs.append(prompt_text)
                        
                        # Add image if exists
                        if uploaded_image:
                            img = Image.open(uploaded_image)
                            inputs.append(img)
                        
                        # Generate
                        response = model.generate_content(inputs)
                        
                        # --- Display Results ---
                        
                        # 1. Metrics Row
                        if label != "N/A":
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("BERT Classification", label.upper())
                            m_col2.metric("Confidence", f"{score:.2%}")
                        
                        # 2. Gemini Analysis
                        st.info(response.text)
                        
                        # 3. Download Report
                        st.download_button(
                            "📥 Download Full Report",
                            data=response.text,
                            file_name="climate_investigation.txt"
                        )

                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

# --- TAB 2: ABOUT ---
with tab2:
    st.header("About this Product")
    st.markdown("""
    **Climate Intelligence Assistant** is a next-generation forensic tool for digital media.
    
    #### ✨ Key Features:
    * **🧠 Hybrid Intelligence:** Combines specialized discriminative AI (ClimateBERT) with Generative Reasoning (Gemini).
    * **👁️ Visual Forensics:** Capable of analyzing misleading graphs, fake memes, and out-of-context photos using Computer Vision.
    * **🕵️ Fallacy Detector:** Goes beyond "True/False" to explain the *manipulation tactics* used (e.g., Cherry Picking data).
    """)