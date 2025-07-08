import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

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
    df = pd.read_csv(input_csv, encoding='utf-8')
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_data('dataset.csv', 'clean_dataset.csv')