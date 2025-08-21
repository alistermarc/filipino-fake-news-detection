import joblib
from flask import Flask, request, render_template
import sklearn
import numpy

# Load the model and the vectorizer
try:
    # Correct path for the best model as per the new README
    model = joblib.load('Models/With Hyperparameter tuning/DecisionTreeClassifier_best_model.joblib')
    # The user must provide this file by saving it from the notebook
    vectorizer = joblib.load('tfidf_vectorizer.joblib') 
except FileNotFoundError:
    # Fallback for different naming or location
    model = None
    vectorizer = None

# Load Tagalog stop words
try:
    with open('tagalog_stop_words.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
except FileNotFoundError:
    stop_words = []

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    text = ""
    error = None

    if not model or not vectorizer:
        error = "Model or vectorizer not found. Please ensure 'DecisionTreeClassifier_best_model.joblib' and 'tfidf_vectorizer.joblib' are in the correct paths."
        return render_template('index.html', error=error)

    if request.method == 'POST':
        text = request.form['text']
        if text:
            # Preprocess the text (simple lowercase and stop word removal)
            processed_text = ' '.join([word for word in text.lower().split() if word not in stop_words])
            
            # Vectorize the text
            vectorized_text = vectorizer.transform([processed_text])
            
            # Make a prediction
            result = model.predict(vectorized_text)
            # The labels are 0 for 'real' and 1 for 'fake'
            prediction = "Fake" if result[0] == 1 else "Real"

    return render_template('index.html', text=text, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
