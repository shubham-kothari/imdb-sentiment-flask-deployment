import json
from flask import Flask, request, jsonify
from imdbSentiment import imdbSentiment
from settings import *

##imdb_sentiment Object
sentiment = imdbSentiment()

#loading volcb, tokenizer, and model
vocab = sentiment.get_vocab(vocab_dir)
tokenizer = sentiment.get_tokenizer(tokenizer_dir)
model = sentiment.get_model(model_dir)
print('Vocab, Tokenizer, and Model is loaded')


app = Flask(__name__)

@app.route('/sentiment', methods=['POST','GET'])
def upload_file():
    if request.method == "GET":
        return "This is the Post API"
    elif request.method == "POST":
        data = request.json
        review, score = sentiment.predict_sentiment_with_probability(data['text'], vocab, tokenizer, model)
        return jsonify({
            "review" : review,
            "score" : score})


## Deploy
if __name__ == "__main__":
    app.run("0.0.0.0",5000)

