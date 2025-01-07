<<<<<<< HEAD
from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load the pre-trained model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "models", "FNDS(LSTM)1.H5")
model = load_model(model_path)

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.num_words = 10000
tokenizer.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer.lower = True

# Load the stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['news_text']

        # Preprocess the input text
        def preprocess_text(text):
            review = re.sub('[^a-zA-Z]', ' ', text)
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if not word in stop_words]
            review = ' '.join(review)
            return review

        preprocessed_text = preprocess_text(text)
        sequences = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequences = pad_sequences(sequences, padding='pre', maxlen=100)

        # Predict fake or real
        prediction = model.predict(padded_sequences)
        result = "Real News" if prediction >= 0.5 else "Fake News"

        return result

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load the pre-trained model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "models", "FNDS(LSTM)1.H5")
model = load_model(model_path)

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.num_words = 10000
tokenizer.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer.lower = True

# Load the stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['news_text']

        # Preprocess the input text
        def preprocess_text(text):
            review = re.sub('[^a-zA-Z]', ' ', text)
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if not word in stop_words]
            review = ' '.join(review)
            return review

        preprocessed_text = preprocess_text(text)
        sequences = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequences = pad_sequences(sequences, padding='pre', maxlen=100)

        # Predict fake or real
        prediction = model.predict(padded_sequences)
        result = "Real News" if prediction >= 0.5 else "Fake News"

        return result

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> c132c2db52035bd2d9ce50d145acffd6b8b7c608
