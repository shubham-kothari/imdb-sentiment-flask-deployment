# imdb-sentiment-flask-deployment
imdb-sentiment-flask-deployment


Requirements:
tensorflow
Keras
nltk
Flask

Request code:
Get Call:
import requests
url = 'http://0.0.0.0:5000/sentiment'
resp = requests.get(url)
resp.text


Post Call:
import requests
url = 'http://0.0.0.0:5000/sentiment'
dic = {"text": 'review'}  
response = requests.post(url, json=dic)
response.json()



Replace your trained files with files inside data folder

