import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset (replace with your actual CSV file)
df = pd.read_csv("your_dataset.csv")  # <-- Update this filename

# Assume last column is the target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

# Save the trained model
with open("pickle/model.pkl", "wb") as f:
    pickle.dump(gbc, f)

print("Model trained and saved to pickle/model.pkl")