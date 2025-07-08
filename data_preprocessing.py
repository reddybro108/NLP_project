import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
<<<<<<< HEAD
=======
import os
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(input_csv, output_csv):
<<<<<<< HEAD
    df = pd.read_csv(input_csv, encoding='utf-8')
=======
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")
    df = pd.read_csv(input_csv, encoding='ISO-8859-1', on_bad_lines='skip', engine='python', nrows=100)
    print("Columns found in CSV:", df.columns.tolist())  # Debug line
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_csv}")

if __name__ == "__main__":
<<<<<<< HEAD
    preprocess_data('dataset.csv', 'clean_dataset.csv')
=======
    preprocess_data('tweets.csv', 'clean_tweets_dataset.csv')
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287
