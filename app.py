
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ------------------ MODEL CONFIG ------------------

MODEL_PATH = 'model.pt'  # Path to your trained BERT + LSTM model
TOKENIZER_NAME = 'GroNLP/hateBERT'  # Pretrained tokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Class map
class_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}

# Define model architecture (must match training)
class HateBERTLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, hidden_dim=128, output_dim=3):
        super(HateBERTLSTMClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        logits = self.classifier(hidden)
        return self.softmax(logits)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HateBERTLSTMClassifier(bert_model_name=TOKENIZER_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------ STREAMLIT APP ------------------

st.set_page_config(page_title="Hate Speech Detection", layout="centered")
st.title("Hate Speech Detection using BERT + LSTM (3-Class)")
st.write("Enter a tweet or comment to classify it as **Hate Speech**, **Offensive**, or **Neutral**.")

# Input field
user_input = st.text_area("🔤 Enter Text:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        encoded = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            padding='max_length',
            max_length=64,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()

        label = class_map[pred_class]
        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence Score:** {confidence:.4f}")
