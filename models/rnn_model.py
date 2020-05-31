from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Activation, Dense
from keras.layers.embeddings import Embedding
from keras import metrics
import numpy as np
import pandas as pd

from database import get_vocab, get_dataframe
import os 
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle




def create_embeddings_indices(glove_file):
	embeddings_indices = dict()
	f = open(glove_file)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_indices[word] = coefs
	f.close()
	return embeddings_indices

def populate_embedding_matrix(glove_file, dim, words, vocabulary_size):
	embeddings_indices = create_embeddings_indices(glove_file)
	embedding_matrix = np.zeros((vocabulary_size, dim))
	for i in range(len(words)):
		embedding_vector = embeddings_indices.get(words[i])
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


def import_vocab(collection_name):
	words, vocabulary_size = get_vocab(collection_name)
	word_dict = {}
	for i in range(len(words)):
		word_dict[words[i]] = i
	return words, word_dict, vocabulary_size

def convert_entry(word_dict, entry):
	tokenized_entry = []
	for word in entry:
		tokenized_entry.append(word_dict[word])
	return tokenized_entry


def format_content(word_dict, dataframe):
	testframe = pd.DataFrame(columns = ['labels', 'word_tokens'])
	for i in range(10):
		for content in dataframe.loc[i].content:
			entry = content[0]
			tokenized_entry = convert_entry(word_dict, entry)
			testframe = testframe.append({'labels': dataframe.loc[i].author, 'word_tokens': tokenized_entry}, ignore_index=True)
	return testframe



#model_conv = create_conv_model()
#model_conv.fit(data, np.array(labels), validation_split=0.4, epochs = 3)

def create_glove_model(vocabulary_size, dim, embedding_matrix):
	## create model
	model_glove = Sequential()
	model_glove.add(Embedding(vocabulary_size, dim, input_length=18, weights=[embedding_matrix], trainable=False))

	model_glove.add(Dense(100, activation = "softmax"))

	model_glove.add(Dropout(0.1, noise_shape=None, seed=None))

	model_glove.add(SimpleRNN(10, activation = 'softmax'))

	model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
	return model_glove


path = os.getcwd()
glove_set = "twitter"
glove_dim = 200


db_collections = ['test_twitter_25_passlen_18_entrymin_250', 'test_twitter_100_passlen_18_entrymin_250', 
'test_wikipedia_300_passlen_18_entrymin_250', 'test_twitter_200_passlen_18_entrymin_250', 
'test_wikipedia_50_passlen_18_entrymin_250', 'test_wikipedia_100_passlen_18_entrymin_250', 
'test_wikipedia_200_passlen_18_entrymin_250', 'test_twitter_50_passlen_18_entrymin_250', 
'forum_data']

if glove_set == "twitter":
	glove_data_file = path + "/glove.twitter.27B/glove.twitter.27B." + str(glove_dim) + "d.txt"
elif glove_set == "wikipedia":
	glove_data_file = path + "/glove.6B/glove.6B." + str(glove_dim) + "d.txt"


words, word_dict, vocabulary_size = import_vocab(db_collections[3])
embedding_matrix = populate_embedding_matrix(glove_data_file, glove_dim, words, vocabulary_size)
raw_dataframe = get_dataframe(db_collections[3])

test_dataframe = format_content(word_dict, raw_dataframe)
test_dataframe = shuffle(test_dataframe)


# Label unique authors
string_labels = test_dataframe['labels'].unique()
label_dict = {}
for i in range(len(string_labels)):
	label_dict[string_labels[i]] = i
labels = test_dataframe['labels'].map(label_dict)
categorical_labels = to_categorical(labels, num_classes=None)


data = np.array(test_dataframe['word_tokens'].tolist())



## Fit train data
glove_model = create_glove_model(vocabulary_size, glove_dim, embedding_matrix)
glove_model.fit(data, categorical_labels, validation_split=0.4, epochs = 1000)
glove_model.summary()

