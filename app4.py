import streamlit as st
from transformers import pipeline
import os
import zipfile
import re
import google.generativeai as genai
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import plotly.graph_objects as go

# --- Auto-Unzip Logic ---
model_folder = 'climatebert_model' 
zip_archive = 'climatebert_model_archive.zip' 

if not os.path.exists(model_folder) and os.path.exists(zip_archive):
    print(f"'{model_folder}' not found. Unzipping '{zip_archive}'...")
    with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("✅ Model unzipped successfully!")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    models = {}
    
    # 1. ClimateBERT (Discriminative)
    if os.path.exists(model_folder):
        try:
            models['bert'] = pipeline("text-classification", model=model_folder, tokenizer=model_folder)
        except Exception as e:
            st.error(f"Error loading ClimateBERT: {e}")
    
    # 2. Sentence Transformer (For Vector Similarity)
    # We use a small, fast model for local embeddings
    try:
        models['similarity'] = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        
    return models

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

# Load Models
models = load_models()
if not models: st.stop()

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🕵️ Quick Check & Forensics", "📄 Document Forensics", "ℹ️ About"])

# --- TAB 1: QUICK CHECK + FORENSICS ---
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("1. Input Data")
        article_title = st.text_input("Headline / Title", placeholder="e.g., New study shows ice is growing...")
        article_text = st.text_area("Article Text", height=200, placeholder="Paste the content here...")
        
        st.markdown("---")
        st.write("📷 **Upload Evidence (Optional)**")
        uploaded_image = st.file_uploader("Upload a chart, meme, or photo", type=["jpg", "png", "jpeg"])
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
                with st.spinner("🤖 Processing... Running BERT, Vector Search, and Gemini..."):
                    try:
                        # --- A. Text Classification (ClimateBERT) ---
                        label = "N/A"
                        score = 0.0
                        combined_text = f"{article_title} {re.sub(r'<.*?>', '', article_text)}"
                        
                        if 'bert' in models and (article_text or article_title):
                            prediction = models['bert'](combined_text[:512])[0] 
                            label = prediction['label']
                            score = prediction['score']

                        # --- B. Scientific Consensus Meter (Vector Math) ---
                        consensus_score = 0.0
                        subjectivity = 0.0
                        
                        if 'similarity' in models and combined_text.strip():
                            # 1. The "Truth Anchor" (Summarized Scientific Consensus)
                            truth_anchor = """
                            Climate change is real and primarily caused by human activities, specifically the emission of greenhouse gases like carbon dioxide. 
                            Global warming leads to rising sea levels, extreme weather events, and loss of biodiversity. 
                            Scientific consensus is overwhelming that urgent action is needed to reduce emissions.
                            """
                            
                            # 2. Encode both texts into vectors
                            embeddings = models['similarity'].encode([truth_anchor, combined_text])
                            
                            # 3. Calculate Cosine Similarity
                            consensus_score = util.cos_sim(embeddings[0], embeddings[1]).item()
                            
                            # 4. TextBlob for Subjectivity (0=Fact, 1=Opinion)
                            blob = TextBlob(combined_text)
                            subjectivity = blob.sentiment.subjectivity

                        # --- C. Display Local Metrics (No API Needed) ---
                        st.markdown("#### 🧪 Forensic Metrics (Local Analysis)")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("BERT Class", label.upper(), f"{score:.2%}")
                        
                        # Display Consensus as a gauge
                        delta_color = "normal"
                        if consensus_score < 0.3: delta_color = "inverse"
                        m_col2.metric("Scientific Alignment", f"{consensus_score:.2%}", delta="Vector Similarity", delta_color=delta_color)
                        
                        # Display Subjectivity
                        sub_label = "Objective (Fact)" if subjectivity < 0.5 else "Subjective (Opinion)"
                        m_col3.metric("Writing Style", sub_label, f"{subjectivity:.2%}")
                        
                        st.divider()

                        # --- D. Gemini Call (Multimodal) ---
                        genai.configure(api_key=gemini_api_key)
                        
                        # Robust connection
                        model = None
                        model_options = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest']
                        for name in model_options:
                            try:
                                test_model = genai.GenerativeModel(name)
                                test_model.generate_content("test")
                                model = test_model
                                break
                            except: continue
                        
                        if model:
                            inputs = []
                            prompt_text = f"""
                            You are an expert Climate Misinformation Analyst.
                            
                            DATA FORENSICS:
                            - BERT Prediction: {label}
                            - Scientific Alignment (Vector Distance): {consensus_score:.2f} (Low = Deviation from consensus)
                            - Subjectivity Score: {subjectivity:.2f} (High = Opinionated)
                            
                            Generate a report:
                            1. **Verdict:** (Start with Emoji).
                            2. **Forensic Interpretation:** Explain the metrics above. (e.g. "The low vector alignment suggests this text deviates significantly from standard scientific definitions").
                            3. **Visual Analysis:** (If image exists).
                            4. **Correction:** Scientific facts.
                            
                            Headline: {article_title}
                            Text: {article_text}
                            """
                            inputs.append(prompt_text)
                            if uploaded_image: inputs.append(Image.open(uploaded_image))
                            
                            response = model.generate_content(inputs)
                            st.subheader("🧠 AI Executive Summary")
                            st.info(response.text)
                            st.download_button("📥 Download Report", response.text, "climate_report.txt")

                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

# --- TAB 2: DOCUMENT FORENSICS (PDF) ---
with tab2:
    st.header("📄 Full Document Forensics")
    st.markdown("Upload a PDF report. We scan the extracted text for manipulation patterns.")
    
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_pdf and st.button("Analyze Document"):
        if not gemini_api_key:
            st.error("Please provide Gemini API Key.")
        else:
            with st.spinner("Processing PDF..."):
                try:
                    reader = PdfReader(uploaded_pdf)
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text() + "\n"
                    
                    preview_text = full_text[:3000] # Analyze first 3000 chars
                    
                    if 'bert' in models:
                        prediction = models['bert'](preview_text[:512])[0]
                        doc_label = prediction['label']
                        doc_score = prediction['score']
                        
                        st.divider()
                        d_col1, d_col2 = st.columns(2)
                        d_col1.metric("Document Class", doc_label.upper())
                        d_col2.metric("Confidence", f"{doc_score:.2%}")

                    genai.configure(api_key=gemini_api_key)
                    model_doc = genai.GenerativeModel('gemini-2.0-flash')
                    doc_prompt = f"""
                    Analyze this extracted text (PDF).
                    Class: {doc_label}.
                    Provide: 1. Document Intent, 2. Scientific Validity, 3. Key Warning Signs.
                    Text: {preview_text}
                    """
                    response_doc = model_doc.generate_content(doc_prompt)
                    
                    st.markdown("### 📝 Executive Summary")
                    st.markdown(response_doc.text)
                    with st.expander("View Extracted Text"):
                        st.text(preview_text)
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 3: ABOUT ---
with tab3:
    st.header("About this Product")
    st.markdown("""
    **Climate Intelligence Assistant** is a hybrid AI forensic tool.
    
    #### 🧬 Technology Stack:
    1.  **ClimateBERT (Local):** Fine-tuned transformer for classification.
    2.  **Sentence-Transformers (Local):** Generates vector embeddings to calculate **Cosine Similarity** against scientific consensus.
    3.  **TextBlob (Local):** Statistical linguistic analysis for subjectivity and polarity.
    4.  **Google Gemini (Cloud):** Generative reasoning and vision capabilities.
    """)