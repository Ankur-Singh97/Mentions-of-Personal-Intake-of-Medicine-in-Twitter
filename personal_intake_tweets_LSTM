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

#converting glove file to word2vec file
glove_input_file = '/home/ankur/Desktop/Internship/glove.twitter.27B.25d.txt'
word2vec_output_file = '/home/ankur/Desktop/Internship/glove.twitter.27B.25d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

#loading the word2vec file
Word2Vec_model = KeyedVectors.load_word2vec_format('/home/ankur/Desktop/Internship/glove.twitter.27B.25d.txt.word2vec', binary = True, unicode_errors = 'ignore')

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
    
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(tweets)
X = tokenizer.texts_to_sequences(tweets)	#converting tweets to sequences of numbers

X = pad_sequences(X,maxlen = 100)	#since keras takes in equal lengthed sequences hence we need to pad the sequence

# Encoding categorical output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)


#one hot encode
labels = to_categorical(labels)



#preparing the LSTM model
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(80, dropout = 0.2))
model.add(Dense(3, activation = 'softmax' ))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(X, labels, nb_epoch=3, batch_size=20)
scores = model.evaluate(X, labels, verbose=0)




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
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_test)
X_test = tokenizer.texts_to_sequences(tweets_test)

X_test = pad_sequences(X_test,maxlen = 100)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
labelencoder = LabelEncoder()
labels_test = labelencoder.fit_transform(labels_test)

# one hot encode
labels_test = to_categorical(labels_test)

# Final evaluation of the model
scores_test = model.evaluate(X_test, labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores_test[1]*100))

