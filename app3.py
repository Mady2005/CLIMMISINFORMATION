import gradio as gr
from transformers import pipeline
import re
import os
import zipfile
from pyngrok import ngrok # Import ngrok

# --- Auto-Unzip Logic for Hugging Face ---
model_folder = 'climatebert_model'
zip_archive = 'climatebert_model_archive.zip'

if not os.path.exists(model_folder) and os.path.exists(zip_archive):
    print(f"'{model_folder}' not found. Unzipping '{zip_archive}'...")
    with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("✅ Model unzipped successfully!")
# ---

print("Loading fine-tuned ClimateBERT model...")
classifier = pipeline("text-classification", model=model_folder)
print("✅ Model loaded successfully!")

def predict_source(title, text):
    if not title and not text: return ""
    combined_text = str(title) + ' ' + re.sub(re.compile('<.*?>'), '', str(text))
    prediction = classifier(combined_text)[0]
    label = prediction['label']
    score = prediction['score']
    return f"Predicted Source: '{label}' (Confidence: {score:.2%})"

iface = gr.Interface(
    fn=predict_source,
    inputs=[gr.Textbox(lines=2, label="Article Title"), gr.Textbox(lines=10, label="Article Text")],
    outputs=gr.Label(label="Analysis Result"),
    title="Climate Misinformation Detector (Powered by ClimateBERT)",
)

if __name__ == "__main__":
    # --- NEW ngrok Integration ---
    # Paste your ngrok authtoken here
    NGROK_AUTHTOKEN = "32zeqmxvWyXLaIEHYv0CVKvOqi4_2tih8YbpAsy5VK8Wi11fG"
    
    
    if NGROK_AUTHTOKEN == "32zeqmxvWyXLaIEHYv0CVKvOqi4_2tih8YbpAsy5VK8Wi11fG":
        ngrok.set_auth_token(NGROK_AUTHTOKEN)
        # Create a public URL tunnel to the Gradio app
        public_url = ngrok.connect(7860) # Gradio's default port is 7860
        print(f"✅ Your public shareable link is: {public_url}")
    else:
        print("--- WARNING: ngrok authtoken is not set. Public URL will not work. ---")
        print("--- Get a free token from https://dashboard.ngrok.com/get-started/your-authtoken ---")
    
    # ---
    
    # Launch the app (without the share=True parameter)
    iface.launch(share=True)