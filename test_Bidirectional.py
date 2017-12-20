#importing the libraries
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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
from keras.layers import Bidirectional

from keras import backend as K


import tensorflow as tf


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(tf.round(y_pred), "float32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = (2 * precision * recall) / (precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

source = '/home/ankur/Desktop/Internship/glove.twitter.27B.25d.txt.word2vec'

#loading the word2vec file
Word2Vec_model = KeyedVectors.load_word2vec_format(source, binary = True, unicode_errors = 'ignore')

#Preparing the word2vec model
embedding_matrix = Word2Vec_model.wv.syn0
embedding_layer = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],input_length=100)


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
    tweets.append(i[4])	#tweets
 

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


X = pad_sequences(y,maxlen = 100)		#since keras takes in equal lengthed sequences hence we need to pad the sequence

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
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128, return_sequences = True, recurrent_dropout = 0, dropout = 0.2)))

model.add(Bidirectional(LSTM(128, recurrent_dropout = 0, dropout = 0.2)))
model.add(Dense(2, activation = 'softmax' ))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = [f1_score, 'accuracy'])
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
