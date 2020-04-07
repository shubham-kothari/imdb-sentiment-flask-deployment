import os
import pickle
from string import punctuation
from nltk.corpus import stopwords

class imdbSentiment:
    
    def clean_doc(self, doc):
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return tokens


    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text


    def get_vocab(self, vocab_filename):
        # vocab_filename = 'vocab.txt'
        vocab = self.load_doc(os.getcwd()+'/'+vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)
        return vocab

    def get_tokenizer(self, dir_tokenizer):
        # load tokenizer
        with open(dir_tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer


    def get_model(self, dir_model):
        # loadin saved model from disk
        model = pickle.load(open(dir_model, 'rb'))
        return model


    # classify a review as negative (0) or positive (1)
    def predict_sentiment(self, review, vocab, tokenizer, model):
        tokens = self.clean_doc(review)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        # convert to line
        line = ' '.join(tokens)
        # encode
        encoded = tokenizer.texts_to_matrix([line], mode='freq')
        # prediction
        yhat = model.predict(encoded, verbose=0)
        return round(yhat[0,0])
    #Rounding won't work well with multi-class classifiers.
    
    
    
    # classify a review as negative (0) or positive (1)
    def predict_sentiment_with_probability(self, review, vocab, tokenizer, model):
        dict_sent_mapping = { 0: 'Negative',
                              1: 'Positive',
                            }
        # clean
        tokens = self.clean_doc(review)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        # convert to line
        line = ' '.join(tokens)
        # encode
        encoded = tokenizer.texts_to_matrix([line], mode='freq')
        # prediction
        yproba = model.predict_proba(encoded, verbose=0)[0][0]    
        # print(yproba)
        #model.predict() and model.predict_proba() gives the same result. For classes, use model.predict_classes()
        yhat = model.predict_classes(encoded, verbose=0)
        sent=dict_sent_mapping[yhat[0][0]] #Using dictionary to reverse mapping of class to label
        #print(sent,yproba)    
        return (sent,str(yproba))  

