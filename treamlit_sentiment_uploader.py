
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from io import StringIO

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# UI
st.title("📊 Sentiment Analysis on Uploaded File")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")

        text_column = st.selectbox("Select the column containing text", df.columns)
        texts = df[text_column].astype(str).tolist()

        # Tokenize and create batches
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        loader = DataLoader(dataset, batch_size=8)

        # Predict
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1)
                preds.extend(batch_preds.cpu().numpy())

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df["Predicted Sentiment"] = [label_map[p] for p in preds]
        st.success("Sentiment analysis complete!")
        st.dataframe(df[[text_column, "Predicted Sentiment"]])

        # Optional download
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_download, "sentiment_results.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
