Climate Intelligence Assistant
An AI-powered web application designed to combat climate misinformation by analyzing the source and content of news articles. This project uses a fine-tuned ClimateBERT model for source classification and integrates Google's Gemini API for advanced, real-time content analysis.



🚀 Core Features
State-of-the-Art Source Classification: Utilizes a ClimateBERT model, fine-tuned on a custom, expanded dataset, to predict the nature of an article's source (science, news, junksci, conspiracy) with high accuracy and a confidence score.

Live Interactive Demo: A user-friendly web interface built with Gradio allows anyone to paste in an article's title and text for instant analysis.

Generative AI-Powered Analysis (via Gemini API): The application goes beyond simple classification by offering advanced features powered by Google's Gemini LLM:

✨ AI Summarization: Generates concise, multi-point summaries of long articles.

✨ Main Claim Identification: Extracts the key arguments and claims from the text.

✨ Bias & Loaded Language Detection: Analyzes the text for potential propaganda techniques, logical fallacies, and emotionally manipulative language.

Automated Data Pipeline: Includes scripts to programmatically fetch new articles from sources like The Guardian and arXiv, enabling continuous improvement of the training dataset.
 Technology Stack
Machine Learning: Python, PyTorch, Hugging Face Transformers, Scikit-learn, Pandas

Core Model: Fine-tuned climatebert/distilroberta-base-climate-f

Generative AI: Google Gemini API

Web Application: Gradio

Development Environment: Google Colab (for GPU-intensive fine-tuning), VS Code

Deployment: Designed for easy deployment on Hugging Face Spaces


⚙️ How It Works
The project follows a complete end-to-end MLOps workflow:

Data Collection: A custom dataset was created by aggregating articles from various sources and expanded using APIs from The Guardian and arXiv.

Model Fine-Tuning: The pre-trained ClimateBERT model was fine-tuned on the expanded dataset in a Google Colab environment to specialize in classifying climate-related text.

Deployment: The final, trained model is hosted on the Hugging Face Hub.

Inference: The Gradio application loads the model from the Hub and serves a user-friendly interface. When a user submits text, the app performs two actions:

It gets a quick classification from the local ClimateBERT model.

It makes live API calls to Google's Gemini for deeper, generative analysis.

## 🚀 Project Demo

Check out a full video walkthrough of this project on YouTube:

[**Watch the Video Demo Here**]## https://youtu.be/Vd7hIjyC0o4
