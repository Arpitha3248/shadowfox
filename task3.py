!pip install transformers datasets torch scikit-learn matplotlib seaborn

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

# Choosing DistilBERT fine-tuned for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
pipeline_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "I love how efficient this model is!",
    "This is a terrible example of customer service.",
    "It works, but I expected more detailed results.",
    "The project was well executed with great teamwork.",
]

results = pipeline_model(texts)

# Print results
for t, r in zip(texts, results):
    print(f"Input: {t}\nPrediction: {r}\n")

dataset = load_dataset("imdb", split="test[:200]")  # Load small subset for speed
true_labels = dataset['label']
sample_texts = dataset['text']

# Predict using LM
pred_labels = []
for text in sample_texts:
    try:
        res = pipeline_model(text[:512])[0]  # Truncate long text
        pred_labels.append(1 if res['label'] == 'POSITIVE' else 0)
    except:
        pred_labels.append(0)

# Accuracy and Report
accuracy = accuracy_score(true_labels, pred_labels)
print(classification_report(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"], zero_division=0))
print("Accuracy on IMDb Subset:", accuracy)
print(report)


import numpy as np

# Collect sample predictions
df_results = pd.DataFrame({
    "True Label": true_labels,
    "Predicted Label": pred_labels
})
df_results["Match"] = df_results["True Label"] == df_results["Predicted Label"]

# Plot
sns.countplot(x="Match", data=df_results)
plt.title("Prediction Match vs Mismatch")
plt.show()

