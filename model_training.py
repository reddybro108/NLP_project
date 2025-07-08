import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(data_csv):
    df = pd.read_csv(data_csv)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model('clean_dataset.csv')
