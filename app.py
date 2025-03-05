from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocess import preprocess_text
from src.similarity_models import tfidf_similarity, bert_similarity

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def predict_similarity(data: TextPair):
    text1 = preprocess_text(data.text1)
    text2 = preprocess_text(data.text2)

    score_tfidf = tfidf_similarity(text1, text2)
    score_bert = bert_similarity(text1, text2)

    return {
        "tfidf_similarity": score_tfidf,
        "bert_similarity": score_bert
    }
