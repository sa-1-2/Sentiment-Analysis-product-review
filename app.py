from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

app = Flask(__name__)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]',' ',text)
    word = word_tokenize(text)
    wnl = WordNetLemmatizer()
    lem_word = [wnl.lemmatize(x) for x in word if x not in punctuation and x not in stopwords.words('english')]
    join_word = " ".join(lem_word)
    return join_word

@app.route('/', methods=['GET', 'POST'])
def home_page():
    result = None
    if request.method == 'POST':
        reviews = request.form.get('review_string')
        reviews = preprocess_text(reviews)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vect = pickle.load(f)
        ans = vect.transform([reviews]).toarray()[0]
        with open("gradient_boosting_classifier.pkl", "rb") as f:
            gbc = pickle.load(f)
        sentiment = gbc.predict([ans])[0]
        if sentiment == 0:
            result = "Negative Sentiment"
        else:
            result = "Positive Sentiment"
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
