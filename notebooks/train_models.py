import os
import sys
import joblib
import json
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from sklearn.utils import shuffle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Add src folder to the Python path
sys.path.append('../src')

# Import custom modules
from data_processing.preprocessing import get_dataset
from data_processing.feature_extraction import prepare_datasets, fit_tfidf_vectorizer, apply_features
from models.simple_models import train_and_evaluate_logreg, train_and_evaluate_mlp, train_and_evaluate_svm
from models.transformer_based_models import load_and_prepare_model, train_and_evaluate
import torch

# Helper function to create timestamped filenames
def get_timestamped_filename(base_name, ext=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ext:
        return f"{base_name}_{timestamp}.{ext}"
    else:
        return f"{base_name}_{timestamp}"

# Paths for saving models
weights_dir = "../src/models/weights"
os.makedirs(weights_dir, exist_ok=True)

# Load datasets
train_dataset, eval_dataset = get_dataset('train', True)
test_dataset = get_dataset('val')


# 1. Fit and save TF-IDF vectorizer
print("Fitting TF-IDF vectorizer...")
texts_train = [f"{input_text} {full_sentence}" for input_text, full_sentence in zip(train_dataset['input'], train_dataset['full_sentence'])]
vectorizer = fit_tfidf_vectorizer(texts_train)
vectorizer_path = os.path.join(weights_dir, get_timestamped_filename("tfidf_vectorizer_final", "joblib"))
joblib.dump(vectorizer, vectorizer_path)
print(f"TF-IDF vectorizer saved to: {vectorizer_path}")

# 2. Generate Features
train_dataset = train_dataset.map(apply_features, batched=True)
eval_dataset = eval_dataset.map(apply_features, batched=True)
validation_dataset = test_dataset.map(apply_features, batched=True)

# 3. Train and save Logistic Regression
print("Training Logistic Regression...")
logreg = train_and_evaluate_logreg(train_dataset, validation_dataset)
logreg_path = os.path.join(weights_dir, get_timestamped_filename("logreg_model_weights_final", "joblib"))
joblib.dump(logreg, logreg_path)
print(f"Logistic Regression model saved to: {logreg_path}")

# 4. Train and save SVM
print("Training SVM...")
svm = train_and_evaluate_svm(train_dataset, validation_dataset)
svm_path = os.path.join(weights_dir, get_timestamped_filename("svm_model_weights_final", "joblib"))
joblib.dump(svm, svm_path)
print(f"SVM model saved to: {svm_path}")

# 5. Train and save MLP
print("Training MLP...")
mlp, scaler  = train_and_evaluate_mlp(train_dataset, validation_dataset)
mlp_path = os.path.join(weights_dir, get_timestamped_filename("mlp_model_weights_final", "joblib"))
joblib.dump(mlp, mlp_path)
print(f"MLP model saved to: {mlp_path}")

scaler_path = os.path.join(weights_dir, get_timestamped_filename("standard_scaler_final", "joblib"))
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# 5. Tokenize datasets for Transformer models
print("Preparing datasets for Transformer models...")
tokenized_train_electra, tokenized_eval_electra, tokenized_test_electra, electra_tokenizer = prepare_datasets(
    train_dataset, eval_dataset, test_dataset, model_type='electra'
)
tokenized_train_roberta, tokenized_eval_roberta, tokenized_test_roberta, roberta_tokenizer = prepare_datasets(
    train_dataset, eval_dataset, test_dataset, model_type='roberta'
)
tokenized_train_bert, tokenized_eval_bert, tokenized_test_bert, bert_tokenizer = prepare_datasets(
    train_dataset, eval_dataset, test_dataset, model_type='bert'
)

# 6. Train and save ELECTRA
print("Training ELECTRA...")
electra_model = load_and_prepare_model('electra')
electra_model, electra_accuracy, electra_f1 = train_and_evaluate(
    model=electra_model,
    tokenizer=electra_tokenizer,
    train_dataset=tokenized_train_electra,
    eval_dataset=tokenized_eval_electra,
    test_dataset=tokenized_test_electra
)
electra_model_path = os.path.join(weights_dir, get_timestamped_filename("electra_model"))
# electra_model.model.save_pretrained(electra_model_path)
electra_tokenizer.save_pretrained(electra_model_path)

# Save the entire model (transformer + classifier)
torch.save(electra_model.model.state_dict(), os.path.join(weights_dir, "electra_model_new.pth"))

# Save the tokenizer as usual
electra_tokenizer.save_pretrained(electra_model_path)

print(f"ELECTRA model and tokenizer saved to: {electra_model_path}")

print(f"ELECTRA model saved to: {electra_model_path}")

# 7. Train and save RoBERTa
print("Training RoBERTa...")
roberta_model = load_and_prepare_model('roberta')
roberta_model, roberta_accuracy, roberta_f1 = train_and_evaluate(
    model=roberta_model,
    tokenizer=roberta_tokenizer,
    train_dataset=tokenized_train_roberta,
    eval_dataset=tokenized_eval_roberta,
    test_dataset=tokenized_test_roberta
)
roberta_model_path = os.path.join(weights_dir, get_timestamped_filename("roberta_model"))
roberta_model.model.save_pretrained(roberta_model_path)
torch.save(roberta_model.model.state_dict(), os.path.join(weights_dir, "roberta_model_new.pth"))
roberta_tokenizer.save_pretrained(roberta_model_path)
print(f"RoBERTa model saved to: {roberta_model_path}")

# 8. Train and save BERT
print("Training BERT...")
bert_model = load_and_prepare_model('bert')
bert_model, bert_accuracy, bert_f1 = train_and_evaluate(
    model=bert_model,
    tokenizer=bert_tokenizer,
    train_dataset=tokenized_train_bert,
    eval_dataset=tokenized_eval_bert,
    test_dataset=tokenized_test_bert
)
bert_model_path = os.path.join(weights_dir, get_timestamped_filename("bert_model"))
bert_model.model.save_pretrained(bert_model_path)
torch.save(bert_model.model.state_dict(), os.path.join(weights_dir, "bert_model_new.pth"))
bert_tokenizer.save_pretrained(bert_model_path)
print(f"BERT model saved to: {bert_model_path}")
