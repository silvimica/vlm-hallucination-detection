import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, output_train_file, output_val_file, test_size=0.2, random_state=42):
    # Load the raw data from JSON
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save the splits as JSON files
    train_df.to_json(output_train_file, orient='records', indent=2)
    val_df.to_json(output_val_file, orient='records', indent=2)

    print(f"Data split complete. Training set saved to {output_train_file} and validation set to {output_val_file}")
