import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_csv):
    df = pd.read_csv(data_csv)
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label'].map(label_map)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline for vectorizer + classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial'))
    ])

    # Hyperparameter grid
    param_grid = {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__min_df': [1, 2],
        'tfidf__max_df': [0.9, 1.0],
        'clf__C': [0.1, 1, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)

    # Evaluate accuracy
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model and vectorizer
    best_tfidf = grid.best_estimator_.named_steps['tfidf']
    best_clf = grid.best_estimator_.named_steps['clf']
    joblib.dump(best_clf, 'sentiment_model.joblib')
    joblib.dump(best_tfidf, 'vectorizer.joblib')
    print("Model and vectorizer trained and saved.")

if __name__ == "__main__":
    train_model('clean_dataset.csv')
