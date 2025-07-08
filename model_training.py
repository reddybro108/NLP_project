import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
=======
from sklearn.metrics import accuracy_score, classification_report
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287
import joblib

def train_model(data_csv):
    df = pd.read_csv(data_csv)
    X = df['clean_text']
    y = df['label']
<<<<<<< HEAD
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
=======
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

if __name__ == "__main__":
    train_model('clean_tweets_dataset.csv')
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287
