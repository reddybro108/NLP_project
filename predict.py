import joblib

def predict_sentiment(text):
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return 'Positive' if prediction[0] == 1 else 'Negative'

if __name__ == "__main__":
    text = input("Enter text for sentiment analysis: ")
    print("Sentiment:", predict_sentiment(text))