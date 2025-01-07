<<<<<<< HEAD
import torch
import os
from flask import Flask, render_template, request
from transformers import BertTokenizerFast
from models import BERT_Arch

app = Flask(__name__)

# Load the pre-trained model and tokenizer
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "models", "bert_model1.pt")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch()

# Load model weights
model.load_state_dict(torch.load(model_path))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['news_text']

        # Tokenize and encode the input text
        tokenized_text = tokenizer.batch_encode_plus([text], max_length=15, pad_to_max_length=True, truncation=True)
        input_ids = torch.tensor(tokenized_text['input_ids'])
        attention_mask = torch.tensor(tokenized_text['attention_mask'])

        # Make a prediction
        with torch.no_grad():
            preds = model(input_ids, attention_mask)
            preds = torch.argmax(preds, dim=1).item()

        result = "Real News" if preds == 0 else "Fake News"

        return result

    return render_template('index12.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
import torch
import os
from flask import Flask, render_template, request
from transformers import BertTokenizerFast
from models import BERT_Arch

app = Flask(__name__)

# Load the pre-trained model and tokenizer
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "models", "bert_model1.pt")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch()

# Load model weights
model.load_state_dict(torch.load(model_path))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['news_text']

        # Tokenize and encode the input text
        tokenized_text = tokenizer.batch_encode_plus([text], max_length=15, pad_to_max_length=True, truncation=True)
        input_ids = torch.tensor(tokenized_text['input_ids'])
        attention_mask = torch.tensor(tokenized_text['attention_mask'])

        # Make a prediction
        with torch.no_grad():
            preds = model(input_ids, attention_mask)
            preds = torch.argmax(preds, dim=1).item()

        result = "Real News" if preds == 0 else "Fake News"

        return result

    return render_template('index12.html')

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> c132c2db52035bd2d9ce50d145acffd6b8b7c608
