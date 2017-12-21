#importing the libraries
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Bidirectional, Input
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.models import Model

from keras import backend as K


import tensorflow as tf


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name = 'kernel', shape = (input_shape[-1],), initializer = 'normal', trainable = True)
       
      
       
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


os.environ['KERAS_BACKEND']='theano'

source = '/home/ankur/Desktop/Internship/glove.twitter.27B.25d.txt.word2vec'

#loading the word2vec file
Word2Vec_model = KeyedVectors.load_word2vec_format(source, binary = True, unicode_errors = 'ignore')

#Preparing the word2vec model
embedding_matrix = Word2Vec_model.wv.syn0
embedding_layer = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],input_length=100, trainable = True)


f = open('/home/ankur/Desktop/Internship/personal_intake_tweets.txt') 

x = f.read()
y = x.split("\n")

data = [] 
tweets = []
labels = []


#preparing the list of tweets and labels
for i in y:
    data.append(i.split("\t")) 
    
for i in data[:-1]:
    labels.append(i[3]) #label for each tweet
    tweets.append(i[4]) #tweets
 

tk = TweetTokenizer()
tknz = []
for i in tweets:
    tknz.append(tk.tokenize(i))



     
tokenizer = Tokenizer()
Y = []

for i in tknz:
    tokenizer.fit_on_texts(i)
    Y.append(tokenizer.texts_to_sequences(i))

y = []
for j in Y:
    y.append([l for i in j for l in i])


X = pad_sequences(y,maxlen = 100)   #since keras takes in equal lengthed sequences hence we need to pad the sequence

# Encoding categorical output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)


#one hot encode
labels = to_categorical(labels)
labels = labels[:,1:]

#Checking our model on the test data
labels_test = []
tweets_test = []
f_test = open('/home/ankur/Desktop/Internship/personal_intake_tweets_dev.txt') 

x_test = f_test.read()
y_test = x_test.split("\n")

data_test = []



for i in y_test:
    data_test.append(i.split("\t")) 
    
for i in data_test[:-1]:
    labels_test.append(i[3])
    tweets_test.append(i[4])
    
tk = TweetTokenizer()
tknz = []
for i in tweets_test:
    tknz.append(tk.tokenize(i))



     
tokenizer = Tokenizer(filters = '!"$%&()*+,-.:;<=>?[\\]^`{|}~\t\n')
Y = []

for i in tknz:
    tokenizer.fit_on_texts(i)
    Y.append(tokenizer.texts_to_sequences(i))

y = []
for j in Y:
    y.append([l for i in j for l in i])


X_test = pad_sequences(y,maxlen = 100)  

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
labelencoder = LabelEncoder()
labels_test = labelencoder.fit_transform(labels_test)

# one hot encode
labels_test = to_categorical(labels_test)
labels_test = labels_test[:,1:]



#preparing the LSTM model
sequence_input = Input(shape = (100,), dtype = 'int32')
embedded_sequences = embedding_layer(sequence_input)
l_LSTM1 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.2))(embedded_sequences)
l_LSTM2 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.2))(l_LSTM1)
l_att = AttLayer()(l_LSTM2)
preds = Dense(2, activation = 'softmax')(l_att)
model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

earlyStopping = EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='auto')
history = model.fit(X, labels, validation_data = (X_test, labels_test), epochs=8, batch_size=100, callbacks = [earlyStopping])
scores = model.evaluate(X, labels, verbose=0)

print("Accuracy of training : %.2f%%" % (scores[1]*100))




y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
labels_test = (labels_test > 0.5)
from sklearn.metrics import accuracy_score
print "Accuracy_sklearn", accuracy_score(labels_test, y_pred)
# Final evaluation of the model
scores_test = model.evaluate(X_test, labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores_test[1]*100))
print "Precision, Recall, FScore ", precision_recall_fscore_support(labels_test, y_pred, average='micro')
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
