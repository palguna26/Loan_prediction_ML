import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('data/train.csv')

# Replace '3+' with 3 in 'Dependents' column
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

# Fill missing values
df.ffill(inplace=True)  # Replaces deprecated method call

# Encode categorical features
cols_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols_to_encode:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as 'loan_model.pkl'")
