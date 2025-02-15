import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Define file paths
DATA_PATH = r"C:\Users\XPS\OneDrive\Desktop\flask\synthetic_creditcard.csv"
SAVE_DIR = r"C:\Users\XPS\OneDrive\Desktop\flask\data"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if 'Class' not in df.columns:
    raise KeyError("The dataset does not contain a 'Class' column.")

# Show class distribution
sns.countplot(x=df['Class'])
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Separate features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset using undersampling
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_scaled, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Save preprocessed data & scaler
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

print("✅ Data Preprocessing Complete. Files saved in:", SAVE_DIR)
