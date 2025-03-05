import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text):
    """
    Clean and preprocess text:
    1. Convert to lowercase
    2. Remove punctuation & special characters
    3. Remove stopwords
    4. Apply lemmatization
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", " ", text)  # Remove punctuation
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return " ".join(words)  # Rejoin words
