
# Hate Speech and Offensive Language Detection

This project presents a comprehensive pipeline for detecting hate speech and offensive language on Twitter using machine learning and deep learning models. The project covers exploratory data analysis (EDA), traditional ML models, BiLSTM with GloVe, a custom Transformer, and a fine-tuned HateBERT + LSTM hybrid model. The final model is deployed using Streamlit for real-time predictions.

## 🔍 Project Structure

- `1. Dataset Analysis and EDA.ipynb`: Data cleaning, visualization, and insights.
- `2. TF-IDF + Logistic Regression Model.ipynb`: Traditional baseline model using TF-IDF.
- `3. Custom Transformer.ipynb`: Transformer model from scratch.
- `4. LSTM Code.ipynb`: BiLSTM using GloVe embeddings.
- `5. BERT + LSTM.ipynb`: Hybrid model using HateBERT + LSTM.
- `requirements.txt`: Environment dependencies.

## 🚀 Features

- Text classification (toxic vs. neutral)
- Balanced class handling using data augmentation
- Deep contextual modeling with BERT-based models
- Real-time web app with Streamlit interface

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

Run any of the Jupyter notebooks to train and evaluate the models. For deployment:

```bash
streamlit run app.py
```

## 📊 Results

The HateBERT + LSTM model achieved 95.1% accuracy and 94.8% macro F1-score, outperforming all other approaches.

## 👨‍💻 Author

Rameez Hassan (30120582) – MSc Artificial Intelligence

## 📄 License

This project is for academic use only.
