from peft import LoraConfig, get_peft_model
from transformers import ElectraForSequenceClassification, RobertaForSequenceClassification,BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Define LoRA configuration
lora_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.01,
    r=32,
    target_modules=["query", "value"],
    bias="none"
)

def load_and_prepare_model(model_name: str, num_labels: int = 2):
    """Load and configure the model with LoRA for sequence classification."""
    if model_name == 'electra':
        model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=num_labels)
    elif model_name == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    elif model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    else:
        raise ValueError("Model name must be either 'electra', 'roberta', or 'bert'.")

    model = get_peft_model(model, lora_config)
    return model

def train_and_evaluate(model, tokenizer, train_dataset, eval_dataset, test_dataset):
    """Train and evaluate the model."""
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=4e-4,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Predict on the test dataset
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions if not isinstance(predictions.predictions, (list, tuple)) else predictions.predictions[1]
    class_preds = np.argmax(logits, axis=1)

    # Calculate evaluation metrics
    true_labels = test_dataset['label']
    accuracy = accuracy_score(true_labels, class_preds)
    f1 = f1_score(true_labels, class_preds, average='macro')

    print(f"{model.config.model_type.capitalize()} Model Accuracy:", accuracy)
    print(f"{model.config.model_type.capitalize()} Model F1 Score (Macro):", f1)

    return trainer, accuracy, f1