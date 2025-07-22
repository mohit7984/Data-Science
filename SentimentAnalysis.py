# Prerequiste--!pip install transformers gradio torch --quiet
from transformers import pipeline
import gradio as gr

# Load sentiment analysis pipeline from HuggingFace
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    # Map SST-2 labels to 3-class sentiment (Neutral is custom logic)
    if label == "NEGATIVE":
        sentiment = "Negative"
    elif label == "POSITIVE":
        sentiment = "Positive"
    else:
        sentiment = "Neutral"
    score = result['score']
    return f"Sentiment: {sentiment}\nConfidence: {score:.2f}"

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter text and get the predicted sentiment using a HuggingFace pre-trained model."
)

demo.launch()
