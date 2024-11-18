from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from scipy.stats import entropy

# Initialize the TF-IDF vectorizer and sentence transformer model
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to fit TF-IDF on the training data
def fit_tfidf_vectorizer(texts):
    tfidf_vectorizer.fit(texts)
    return tfidf_vectorizer

# Feature extraction function
def extract_features(texts, vectorizer, include_ttr=True, include_subjectivity=True, include_entropy=True, fit_vectorizer=False):
    # Transform texts with TF-IDF
    tfidf_features = vectorizer.transform(texts).toarray()  # Shape: (num_texts, num_features)
    
    # Calculate type-token ratio (TTR)
    if include_ttr:
        ttr_features = np.array([len(set(word_tokenize(text))) / len(word_tokenize(text)) for text in texts]).reshape(-1, 1)
    else:
        ttr_features = np.zeros((len(texts), 1))  # Dummy array to maintain shape consistency

    # Other feature calculations
    word_counts = np.array([len(word_tokenize(text)) for text in texts]).reshape(-1, 1)
    char_counts = np.array([len(text) for text in texts]).reshape(-1, 1)
    avg_word_len = np.array([np.mean([len(word) for word in word_tokenize(text)]) for text in texts]).reshape(-1, 1)

    # Sentiment Analysis
    polarity_scores = np.array([TextBlob(text).sentiment.polarity for text in texts]).reshape(-1, 1)
    
    if include_subjectivity:
        subjectivity_scores = np.array([TextBlob(text).sentiment.subjectivity for text in texts]).reshape(-1, 1)
    else:
        subjectivity_scores = np.zeros((len(texts), 1))  # Dummy array to maintain shape consistency

    # Calculate text entropy
    if include_entropy:
        entropy_features = np.array([entropy([text.count(char) / len(text) for char in set(text)]) for text in texts]).reshape(-1, 1)
    else:
        entropy_features = np.zeros((len(texts), 1))  # Dummy array to maintain shape consistency

    # Concatenate all features along the last axis
    features = np.concatenate([
        tfidf_features,
        ttr_features,
        word_counts,
        char_counts,
        avg_word_len,
        polarity_scores,
        subjectivity_scores,
        entropy_features
    ], axis=1)
    
    return {"features": features}

# Apply feature extraction to each batch in the dataset
def apply_features(batch, tf_idf=tfidf_vectorizer):
    return extract_features(
        texts=batch['input_output'],
        vectorizer=tf_idf,
        include_ttr=True,
        include_subjectivity=True,
        include_entropy=False,
        fit_vectorizer=False
    )

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
from transformers import ElectraTokenizer, RobertaTokenizer, BertTokenizer
from datasets import Dataset

electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_electra(batch, max_length=512):
    """Tokenizes data for the Electra model."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for input_text, output_text, label in zip(batch['input'], batch['full_sentence'], batch['label']):
        prompt_tokens = electra_tokenizer(input_text, max_length=252, truncation=True, add_special_tokens=False)['input_ids']
        answer_tokens = electra_tokenizer(output_text, max_length=256, truncation=True, add_special_tokens=False)['input_ids']
        
        combined_tokens = [electra_tokenizer.cls_token_id] + prompt_tokens + [electra_tokenizer.sep_token_id] + answer_tokens + [electra_tokenizer.sep_token_id]
        attention_mask = [1] * len(combined_tokens)
        
        padding_length = max_length - len(combined_tokens)
        combined_tokens += [electra_tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        input_ids_list.append(combined_tokens)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)

    return {'input_ids': input_ids_list, 'attention_mask': attention_mask_list, 'labels': labels_list}


def tokenize_roberta(batch, max_length=512):
    """Tokenizes data for the RoBERTa model."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for input_text, output_text, label in zip(batch['input'], batch['full_sentence'], batch['label']):
        prompt_tokens = roberta_tokenizer(input_text, max_length=252, truncation=True, add_special_tokens=False)['input_ids']
        answer_tokens = roberta_tokenizer(output_text, max_length=256, truncation=True, add_special_tokens=False)['input_ids']
        
        combined_tokens = [roberta_tokenizer.cls_token_id] + prompt_tokens + [roberta_tokenizer.sep_token_id] + answer_tokens + [roberta_tokenizer.sep_token_id]
        attention_mask = [1] * len(combined_tokens)
        
        padding_length = max_length - len(combined_tokens)
        combined_tokens += [roberta_tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        input_ids_list.append(combined_tokens)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)

    return {'input_ids': input_ids_list, 'attention_mask': attention_mask_list, 'labels': labels_list}

def tokenize_bert(batch, max_length=512):
    """Tokenizes data for the BERT model."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for input_text, output_text, label in zip(batch['input'], batch['full_sentence'], batch['label']):
        prompt_tokens = bert_tokenizer(input_text, max_length=252, truncation=True, add_special_tokens=False)['input_ids']
        answer_tokens = bert_tokenizer(output_text, max_length=256, truncation=True, add_special_tokens=False)['input_ids']
        
        combined_tokens = [bert_tokenizer.cls_token_id] + prompt_tokens + [bert_tokenizer.sep_token_id] + answer_tokens + [bert_tokenizer.sep_token_id]
        attention_mask = [1] * len(combined_tokens)
        
        padding_length = max_length - len(combined_tokens)
        combined_tokens += [bert_tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        input_ids_list.append(combined_tokens)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)

    return {'input_ids': input_ids_list, 'attention_mask': attention_mask_list, 'labels': labels_list}


def prepare_datasets(train_dataset, eval_dataset, test_dataset, model_type='electra'):
    """Applies the specified tokenizer function to the train, eval, and test datasets and returns the tokenizer."""
    
    if model_type == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        tokenized_train_dataset = train_dataset.map(tokenize_electra, batched=True, keep_in_memory=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_electra, batched=True, keep_in_memory=True)
        tokenized_test_dataset = test_dataset.map(tokenize_electra, batched=True, keep_in_memory=True)
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenized_train_dataset = train_dataset.map(tokenize_roberta, batched=True, keep_in_memory=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_roberta, batched=True, keep_in_memory=True)
        tokenized_test_dataset = test_dataset.map(tokenize_roberta, batched=True, keep_in_memory=True)
    elif model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_train_dataset = train_dataset.map(tokenize_bert, batched=True, keep_in_memory=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_bert, batched=True, keep_in_memory=True)
        tokenized_test_dataset = test_dataset.map(tokenize_bert, batched=True, keep_in_memory=True)
    else:
        raise ValueError("model_type should be either 'electra', 'roberta', or 'bert'")
    
    return tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset, tokenizer


def prepare_single_dataset(dataset, model_type='electra'):
    """Applies the specified tokenizer function to a single dataset and returns the tokenized dataset and tokenizer."""
    
    if model_type == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        tokenized_dataset = dataset.map(tokenize_electra, batched=True, keep_in_memory=True)
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenized_dataset = dataset.map(tokenize_roberta, batched=True, keep_in_memory=True)
    elif model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_dataset = dataset.map(tokenize_bert, batched=True, keep_in_memory=True)
    else:
        raise ValueError("model_type should be either 'electra', 'roberta', or 'bert'")
    
    return tokenized_dataset
