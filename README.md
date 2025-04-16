# streamlit_sentiment_uploader
# 🤖 BERT-Based Sentiment Analysis Web App

A Streamlit-powered app that allows users to upload a CSV or Excel file and perform sentiment analysis using a fine-tuned BERT model.

## 🚀 Features
- Upload CSV/XLSX files
- Choose the column containing text
- Predict sentiment (Positive, Neutral, Negative)
- View results in-app
- Download results as CSV

## 🛠 Setup Instructions

```bash
git clone https://github.com/yourusername/bert-sentiment-app.git
cd bert-sentiment-app/app

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_sentiment_uploader.py
