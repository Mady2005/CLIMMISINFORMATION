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
st.markdown("### AI-Powered Detection & Explanation of Climate Misinformation")

# Load Model
classifier = load_classifier()
if classifier is None: st.stop()

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🕵️ Analyze Content", "ℹ️ About the Project"])

# --- TAB 1: ANALYSIS (Main Tool) ---
with tab1:
    st.write("Paste a news article below to check its credibility.")
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        article_title = st.text_input("Article Title", placeholder="e.g., Scientists claim...")
    with col_input2:
        article_text = st.text_area("Article Text", height=250, placeholder="Paste the full content here...")

    analyze_btn = st.button("🔍 Analyze Credibility", type="primary", use_container_width=True)

    if analyze_btn:
        if not article_title and not article_text:
            st.warning("⚠️ Please enter a title or text to analyze.")
        elif not gemini_api_key:
            st.error("🔑 Please enter your Gemini API Key in the sidebar.")
        else:
            with st.spinner("🤖 Analyzing... Checking sources, detecting tone, and generating fact-checks..."):
                try:
                    # 1. Classification
                    combined_text = f"{article_title} {re.sub(r'<.*?>', '', article_text)}"
                    prediction = classifier(combined_text)[0]
                    label = prediction['label']
                    score = prediction['score']

                    # Display Results
                    st.divider()
                    st.subheader("📊 Analysis Dashboard")
                    
                    # Metric Cards
                    r_col1, r_col2, r_col3 = st.columns(3)
                    r_col1.metric("Predicted Source", label.upper(), delta="ClimateBERT Model")
                    r_col2.metric("Confidence Score", f"{score:.2%}")
                    
                    if label.lower() in ['junksci', 'conspiracy']:
                        r_col3.error("⚠️ High Risk of Misinformation")
                    else:
                        r_col3.success("✅ Likely Credible Source")

                    # 2. Gemini Call (Advanced Prompting)
                    genai.configure(api_key=gemini_api_key)
                    model = None
                    model_options = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest']
                    
                    for name in model_options:
                        try:
                            test_model = genai.GenerativeModel(name)
                            test_model.generate_content("Hello") 
                            model = test_model
                            break 
                        except:
                            continue
                    
                    if model is None:
                        st.error("Could not connect to Gemini. Please check API Key.")
                    else:
                        # --- NEW: Advanced Prompt for More Features ---
                        prompt = f"""
                        The article is classified as '{label}' ({score:.0%} confidence).
                        
                        Please provide a response with exactly these three sections:
                        
                        1. **Tone Analysis:** (Is it Alarmist, Neutral, Scientific, or Aggressive? Explain in 1 sentence).
                        2. **Explanation:** (Explain WHY it was classified as {label} in simple Hinglish).
                        3. **Fact-Check Queries:** (Give 2 bullet points of what the user should Google to verify this information).
                        
                        Title: {article_title}
                        Text: {article_text}
                        """
                        response = model.generate_content(prompt)
                        
                        st.info(response.text)
                        
                        # --- NEW: Download Report Button ---
                        report_text = f"""
                        CLIMATE INTELLIGENCE REPORT
                        ---------------------------
                        Title: {article_title}
                        Predicted Source: {label}
                        Confidence: {score:.2%}
                        
                        AI Analysis:
                        {response.text}
                        """
                        st.download_button(
                            label="📥 Download Analysis Report",
                            data=report_text,
                            file_name="climate_analysis_report.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- TAB 2: ABOUT (Project Info) ---
with tab2:
    st.header("About this Project")
    st.markdown("""
    **Climate Intelligence Assistant** is a hybrid AI tool designed to combat climate change misinformation.
    
    #### 🛠️ How it Works:
    1.  ** Discriminative AI (ClimateBERT):** * We fine-tuned a `distilroberta-base` model specifically on climate data.
        * It classifies text into 4 categories: `Scientific`, `News`, `Junk Science`, `Conspiracy`.
        * Achieved **100% Accuracy** on our validation set.
    
    2.  ** Generative AI (Google Gemini):**
        * The classification label is sent to Google's Gemini 2.0 Flash model.
        * It generates a human-readable explanation in **Hinglish** to help users understand *why* an article is fake or real.
        
    #### 🎓 Created By:
    * Department of Computer Science & Engineering (AI & DS)
    * **Shree Siddheshwar Women’s College of Engineering, Solapur**
    """)