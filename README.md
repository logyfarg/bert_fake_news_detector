# 📰 Fake News Detector with BERT

This project uses the BERT transformer model to classify whether a news article is REAL ✅ or FAKE ❌.

## 🔍 How It Works
- Preprocesses the text using BERT tokenizer
- Fine-tunes a pre-trained BERT model (`bert-base-uncased`) using Hugging Face Transformers
- Evaluates and predicts new headlines interactively

## 📁 Files
- `fake_news_bert.py` – the main script for training and inference
- `requirements.txt` – install dependencies
- `model/` – (optional) saved model weights
- `data/` – (optional) dataset folder

## 🚀 Try It Out
You can train the model with:

```bash
python fake_news_bert.py
🤖 Tech Stack
Python

PyTorch

Hugging Face Transformers

BERT
