import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import numpy as np
import json

# Load the datasets
train_df = pd.read_csv('Training Dataset.csv')
test_df = pd.read_csv('Test Dataset.csv')

# Combine train and test datasets to preprocess them together
test_df['Loan_Status'] = np.nan  # Add dummy target variable to test set for consistent processing
df = pd.concat([train_df, test_df], ignore_index=True)

# Handle missing values (if necessary)
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Education': df['Education'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].mean(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].mean(),
    'Credit_History': df['Credit_History'].mean(),
    'Property_Area': df['Property_Area'].mode()[0]
}, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'].astype(str))
df['Married'] = label_encoder.fit_transform(df['Married'].astype(str))
df['Dependents'] = label_encoder.fit_transform(df['Dependents'].astype(str))
df['Education'] = label_encoder.fit_transform(df['Education'].astype(str))
df['Self_Employed'] = label_encoder.fit_transform(df['Self_Employed'].astype(str))
df['Property_Area'] = label_encoder.fit_transform(df['Property_Area'].astype(str))

# Encode the target variable
df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'].astype(str))

# Separate features and target variable from the training data only
X = df.loc[:len(train_df)-1, df.columns != 'Loan_Status'].drop(columns=['Loan_ID'])
y = df.loc[:len(train_df)-1, 'Loan_Status'].astype(int)

# Split the training data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('loan_approval_model.keras')

# Save the scaler configuration
scaler_filename = "scaler.json"
scaler_data = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}
with open(scaler_filename, 'w', encoding='utf-8') as f:
    f.write(json.dumps(scaler_data, ensure_ascii=False))
