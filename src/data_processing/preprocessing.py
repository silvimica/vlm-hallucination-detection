import json
import pandas as pd
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os 

def get_dataset(dataset, split=False):
    
    os.makedirs('../data/processed', exist_ok=True)
    with open(f'../data/raw/{dataset}_raw.json', 'r') as file:
        data = json.load(file)
    parsed = []
    # Extract input and output at the sentence level
    for el in data:
        input_text = el['question'].replace('<image>', '')
        input_text = input_text.replace('\n', '')

        response_text = el['response']

        # Split the full response into sentences
        sentences = nltk.sent_tokenize(response_text)

        # Initialize sentence labels as ACC by default
        sentence_labels = {sentence: 'ACC' for sentence in sentences}

        # Check each annotation and see if it marks the sentence as INACCURATE
        for annotation in el['annotations']:
            annotation_text = response_text[annotation['start']:annotation['end']]
            annotation_label = annotation['label']
            
            if annotation_label == "INACCURATE":
                # If annotation is INACCURATE, find the sentence it belongs to
                for sentence in sentences:
                    if annotation_text in sentence:
                        sentence_labels[sentence] = 'INACC'  # Mark the sentence as INACCURATE

        # Append each sentence along with its label and other details
        for sentence, label in sentence_labels.items():
            parsed.append({
                'input': input_text,
                'response': response_text,
                'full_sentence': sentence,
                'input_output': f"{input_text} {sentence}",
                'label': 1 if label =='INACC' else 0 # ACC or INACC
            })

    # Create DataFrame
    df = pd.DataFrame(parsed)
    
    if split:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset_split = Dataset.from_pandas(train_df)
        validation_dataset_split = Dataset.from_pandas(val_df)
        train_df.to_json('../data/processed/train_split_processed.json', orient='records', indent=2)
        val_df.to_json('../data/processed/test_split_processed.json', orient='records', indent=2)
        return train_dataset_split, validation_dataset_split
        
    df.to_json(f'../data/processed/val_processed.json', orient='records', indent=2)
    # Convert DataFrame to Hugging Face Dataset format
    dataset = Dataset.from_pandas(df)
    return dataset