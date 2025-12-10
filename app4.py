import streamlit as st
from transformers import pipeline
import os
import zipfile
import re
import google.generativeai as genai
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
import pandas as pd
import plotly.express as px

# --- Auto-Unzip Logic ---
model_folder = 'climatebert_model' 
zip_archive = 'climatebert_model_archive.zip' 

if not os.path.exists(model_folder) and os.path.exists(zip_archive):
    with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
        zip_ref.extractall('.')

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists(model_folder):
            models['bert'] = pipeline("text-classification", model=model_folder, tokenizer=model_folder)
        models['similarity'] = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e: st.error(f"Model Error: {e}")
    return models

# --- Helper: Agentic Search ---
def search_web_for_verification(query):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(f"climate change fact check {query}", max_results=2))
    except: return []

# --- Page Config ---
st.set_page_config(page_title="Climate Intelligence Assistant", page_icon="🌍", layout="wide")

# --- Sidebar ---
st.sidebar.header("⚙️ Enterprise Config")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.title("🌍 Climate Intelligence Assistant")
st.markdown("### 🛡️ Enterprise Risk & Compliance Platform")

models = load_models()
if not models: st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🕵️ Quick Check", "📄 Doc Forensics", "📊 Bulk Audit (Enterprise)", "ℹ️ About"])

# --- TAB 1: QUICK CHECK (Agentic) ---
with tab1:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Input Data")
        article_text = st.text_area("Content to Analyze", height=150)
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    with col_right:
        st.subheader("Forensic Report")
        if st.button("🚀 Analyze Single Item", type="primary"):
            if not gemini_api_key: st.error("Need API Key")
            else:
                with st.spinner("Running Agentic Analysis..."):
                    try:
                        # 1. BERT
                        label, score = "N/A", 0.0
                        if 'bert' in models:
                            pred = models['bert'](article_text[:512])[0]
                            label, score = pred['label'], pred['score']
                        
                        # 2. Consensus Meter
                        consensus = 0.0
                        if 'similarity' in models:
                            anchor = "Climate change is real, human-caused, and driven by CO2."
                            emb = models['similarity'].encode([anchor, article_text])
                            consensus = util.cos_sim(emb[0], emb[1]).item()

                        # 3. Live Web Search
                        web_data = search_web_for_verification(article_text[:50])
                        
                        # Display
                        c1, c2 = st.columns(2)
                        c1.metric("BERT Class", label.upper(), f"{score:.2%}")
                        c2.metric("Scientific Align", f"{consensus:.2%}", delta_color="normal" if consensus>0.5 else "inverse")
                        
                        # 4. Gemini Synthesis
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        web_context = "\n".join([f"- {r['title']}: {r['body']}" for r in web_data])
                        inputs = [f"Analyze. Class: {label}. Consensus: {consensus:.2f}. Web Data: {web_context}. Text: {article_text}"]
                        if uploaded_image: inputs.append(Image.open(uploaded_image))
                        
                        st.info(model.generate_content(inputs).text)
                        
                    except Exception as e: st.error(f"Error: {e}")

# --- TAB 2: DOC FORENSICS ---
with tab2:
    st.header("📄 Corporate Audit (PDF)")
    pdf_file = st.file_uploader("Upload Report", type=["pdf"])
    if pdf_file and st.button("Audit PDF"):
        with st.spinner("Scanning..."):
            reader = PdfReader(pdf_file)
            text = "".join([p.extract_text() for p in reader.pages])[:4000]
            
            # Fast Check
            if 'bert' in models:
                pred = models['bert'](text[:512])[0]
                st.metric("Document Risk Class", pred['label'])
            
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            st.write(model.generate_content(f"Audit for Greenwashing. Text: {text}").text)

# --- TAB 3: BULK AUDIT (NEW ENTERPRISE FEATURE) ---
with tab3:
    st.header("📊 Bulk Compliance Audit")
    st.markdown("Upload a CSV file (column name: 'text') to audit hundreds of claims instantly.")
    
    # 1. Download Sample CSV
    sample_data = pd.DataFrame({"text": ["Solar activity causes warming, not CO2.", "Carbon emissions reached record highs in 2024.", "Wind turbines kill all birds."]})
    st.download_button("📥 Download Sample CSV", sample_data.to_csv(index=False), "sample_audit.csv")
    
    # 2. Upload CSV
    uploaded_csv = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        if 'text' not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            if st.button("🚀 Run Batch Audit"):
                with st.spinner(f"Auditing {len(df)} rows..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    truth_anchor = "Climate change is real, human-caused, and driven by CO2."
                    
                    for i, row in df.iterrows():
                        text = str(row['text'])
                        
                        # Run BERT
                        bert_label = "Error"
                        if 'bert' in models:
                            pred = models['bert'](text[:512])[0]
                            bert_label = pred['label']
                            
                        # Run Consensus
                        consensus = 0.0
                        if 'similarity' in models:
                            emb = models['similarity'].encode([truth_anchor, text])
                            consensus = util.cos_sim(emb[0], emb[1]).item()
                        
                        # Determine Risk
                        risk = "LOW"
                        if bert_label in ['junksci', 'conspiracy'] or consensus < 0.4:
                            risk = "HIGH"
                            
                        results.append({
                            "Original Text": text,
                            "BERT Class": bert_label,
                            "Scientific Score": round(consensus, 2),
                            "Risk Level": risk
                        })
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Create Results DF
                    results_df = pd.DataFrame(results)
                    
                    # Display Stats
                    st.success("Audit Complete!")
                    
                    # Visuals
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Total Items Audited", len(results_df))
                    with m2:
                        high_risk_count = len(results_df[results_df['Risk Level'] == "HIGH"])
                        st.metric("High Risk Items Found", high_risk_count, delta="Requires Attention", delta_color="inverse")
                    
                    # Chart
                    fig = px.pie(results_df, names='Risk Level', title='Audit Risk Distribution', color='Risk Level', color_discrete_map={'HIGH':'red', 'LOW':'green'})
                    st.plotly_chart(fig)
                    
                    # Show Data
                    st.dataframe(results_df)
                    
                    # Download Report
                    st.download_button(
                        "📥 Download Audit Report (CSV)",
                        results_df.to_csv(index=False),
                        "audit_report.csv"
                    )

# --- TAB 4: ABOUT ---
with tab4:
    st.header("About")
    st.write("Enterprise Forensic Tool powered by ClimateBERT & Gemini.")