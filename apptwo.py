import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import os

# --- 1. Load the Fine-Tuned Model ---
# This is the most important step. It loads the "brain" you trained in Colab.
model_path = "climatebert_model"
print("Loading fine-tuned ClimateBERT model...")

# Check if the model directory exists to provide a helpful error message.
if not os.path.isdir(model_path):
    print("--- FATAL ERROR ---")
    print(f"Model directory not found at: '{model_path}'")
    print("Please make sure you have downloaded, unzipped, and correctly named the model folder.")
    print("--------------------")
    # We exit here because the app cannot run without the model.
    exit()

# Load the tokenizer and the model from the saved directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a text classification pipeline.
# This pipeline handles all the complex steps of tokenization and prediction for us.
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

print("✅ Model loaded successfully!")

# --- 2. Create the Prediction Function ---
# This function will be called every time a user clicks "Submit" in the Gradio app.
def predict_source(title, text):
    """
    Takes a title and text, combines them, and returns the model's prediction.
    """
    # Simple validation: if both inputs are empty, return an empty string.
    if not title and not text:
        return ""

    # Combine the title and text for a more complete analysis.
    # Also, clean any potential HTML tags from the text.
    combined_text = str(title) + ' ' + re.sub(re.compile('<.*?>'), '', str(text))

    # The pipeline returns a list of dictionaries, e.g., [{'label': 'news', 'score': 0.99}]
    prediction = classifier(combined_text)[0]

    # Extract the label (predicted source) and score (model's confidence)
    label = prediction['label']
    score = prediction['score']

    # Format the output string for the user
    return f"Predicted Source: '{label}' (Confidence: {score:.2%})"


# --- 3. Create the Gradio Interface ---
# This section defines the look and feel of your web application.
print("Launching Gradio interface...")

iface = gr.Interface(
    fn=predict_source, # The function to call when the user interacts
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter article title here...", label="Article Title"),
        gr.Textbox(lines=10, placeholder="Paste article text here...", label="Article Text")
    ],
    outputs=gr.Label(label="Analysis Result"), # A simple text label for the output
    title="Climate Misinformation Detector (Powered by ClimateBERT)",
    description="Enter the title and text of a climate-related article to predict its source. This demo uses a fine-tuned ClimateBERT model to analyze the content.",
    examples=[
        [
            "Groundbreaking Study Shows Solar Panels Now Work at Night",
            "<p>In an astonishing leap for renewable energy, scientists at the Global Energy Institute have unveiled a new type of photovoltaic cell that generates electricity from moonlight. The study was published in the journal 'Radical Science Tomorrow'.</p>"
        ],
        [
            "Climate change is killing fireflies – threatening a US summer ritual",
            "<p>Max Vogel, a 29-year-old public defense attorney, was picnicking with friends in early August at Prospect Park in Brooklyn, New York, when he noticed flashes of light appear in the air around him. They were fireflies, bioluminescent insects that the Washington DC native had not seen while living in Oregon, where there are few, if any.</p>"
        ]
    ]
)

# --- 4. Launch the App ---
if __name__ == "__main__":
    iface.launch()

