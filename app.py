import joblib
from flask import Flask, request, render_template
import sklearn
import numpy

# Load the model
try:
    model = joblib.load('Models/With_Hyperparameter_Tuning/SGDClassifier_best_model.joblib')
except FileNotFoundError:
    model = None

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

    if not model:
        error = "Model not found. Please ensure 'SGDClassifier_best_model.joblib' is in the correct path."
        return render_template('index.html', error=error)

    if request.method == 'POST':
        text = request.form['text']
        if text:
            # Preprocess the text (simple lowercase and stop word removal)
            processed_text = ' '.join([word for word in text.lower().split() if word not in stop_words])
            
            # Make a prediction
            result = model.predict([processed_text])
            # The labels are 0 for 'real' and 1 for 'fake'
            prediction = "Fake" if result[0] == 1 else "Real"

    return render_template('index.html', text=text, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
