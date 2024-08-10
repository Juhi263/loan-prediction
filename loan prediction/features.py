import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Load the training dataset
train_df = pd.read_csv('Training Dataset.csv')

# Define features to encode
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Initialize and fit label encoders, then save them as JSON files
for feature in categorical_features:
    encoder = LabelEncoder()
    encoder.fit(train_df[feature].astype(str))
    
    # Create dictionary to save classes
    encoder_data = {'classes': encoder.classes_.tolist()}
    
    # Save encoder as JSON file
    with open(f'{feature}_encoder.json', 'w') as f:
        json.dump(encoder_data, f, indent=4)

print("Label encoders saved successfully.")
