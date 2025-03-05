from fastapi import FastAPI
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.utils.vectorization import get_tfidf_vectors

app = FastAPI()

# ✅ Corrected File Path
ml_model_path = r"text_similarity_ml.pkl"
ml_model = joblib.load(ml_model_path)

# Load Transformer Model
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/")
def home():
    return {"message": "Text Similarity API is running 🚀"}

@app.post("/predict/ml")
def predict_ml(text1: str, text2: str):
    """ Predict similarity using ML Model """
    vector = get_tfidf_vectors([text1], [text2]).reshape(-1, 1)
    similarity_score = ml_model.predict(vector)[0]
    return {"similarity_score": similarity_score}

@app.post("/predict/transformer")
def predict_transformer(text1: str, text2: str):
    """ Predict similarity using Transformer Model """
    embedding1 = transformer_model.encode(text1, convert_to_tensor=True)
    embedding2 = transformer_model.encode(text2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return {"similarity_score": similarity_score}
