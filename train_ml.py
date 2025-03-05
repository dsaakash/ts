import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.vectorization import get_tfidf_vectors

# ✅ Load data
data_path = r"Dataset/preprocessed_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"🚨 Data file not found at {data_path}")

df = pd.read_csv(data_path)

# ✅ Convert text to vectors
X = get_tfidf_vectors(df["text1"].tolist(), df["text2"].tolist())
y = np.random.rand(len(X))  # Placeholder similarity labels (Replace with actual labels if available)

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
print(f"📉 Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# ✅ Ensure model save directory exists
model_dir = r"C:/Users/savan/Downloads/DataNeuron_DataScience_Task1/text_similarity_project/src/models/"
os.makedirs(model_dir, exist_ok=True)  # Creates all missing folders

# ✅ Define model path
model_path = os.path.join(model_dir, "text_similarity_ml.pkl")

# ✅ Save the trained model
joblib.dump(model, model_path)
print(f"✅ Model saved successfully at: {model_path}")
