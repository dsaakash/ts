from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv(r"C:/Users/savan/Downloads/DataNeuron_DataScience_Task1/Dataset/preprocessed_data.csv")

# Convert text to embeddings
embeddings1 = model.encode(df["text1"].tolist(), convert_to_tensor=True)
embeddings2 = model.encode(df["text2"].tolist(), convert_to_tensor=True)

# Compute similarity
cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2).numpy().diagonal()

# Save similarity scores
df["similarity"] = cosine_similarities
df.to_csv(r"text_similarity_project\predictions.csv", index=False)
print("✅ Similarity predictions saved in 'predictions.csv'")
