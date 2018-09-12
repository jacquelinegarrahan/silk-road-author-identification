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
from sklearn.model_selection import StratifiedKFold


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


def generate_subsets(data, k, categorical_labels):
	validation_sets = []
	training_sets = []
	valid_target = []
	train_target =[]
	kf = StratifiedKFold(k)
	X = [entry for entry in data]
	y = [entry for entry in categorical_labels]
	for train_index, test_index in kf.split(X, y):
		train =[]
		train_targ = []
		val_targ = []
		val = []

		for index in test_index:
			val.append(X[index])
			val_targ.append(y[index])
		for oindex in train_index:
			train.append(X[oindex])
			train_targ.append(y[oindex])
		training_sets.append(train)
		train_target.append(to_categorical(train_targ))
		validation_sets.append(val)
		valid_target.append(to_categorical(val_targ))


	return validation_sets, valid_target, training_sets, train_target

#model_conv = create_conv_model()
#model_conv.fit(data, np.array(labels), validation_split=0.4, epochs = 3)

def create_glove_model(vocabulary_size, dim, embedding_matrix, activation, loss, optimizer, hidden_layer_size, dropout):
	## create model
	model_glove = Sequential()

	model_glove.add(Embedding(vocabulary_size, dim, input_length=18, weights=[embedding_matrix], trainable=False))
	
	#main layer
#	model_glove.add(Dense(50, activation='softmax', input_shape = (4283,)))

	#hidden layers
	#model_glove.add(Dropout(0.2, noise_shape=None, seed=None))
	model_glove.add(Dense(hidden_layer_size, activation = activation))

	model_glove.add(Dropout(dropout, noise_shape=None, seed=None))
#	model_glove.add(Conv1D(50, 18, activation='softmax'))
	#model_glove.add(MaxPooling1D(pool_size=2))
	model_glove.add(SimpleRNN(10, activation = activation))
	#model_glove.add(Dropout(0.2, noise_shape=None, seed=None))
	#model_glove.add(Dense(10, activation='softmax'))
	model_glove.compile(loss=loss, optimizer=optimizer, metrics=[metrics.categorical_accuracy])
	return model_glove


def create_dataframe(glove_set, glove_dim):
	db_collections = ['test_twitter_25_passlen_18_entrymin_250', 'test_twitter_100_passlen_18_entrymin_250', 
	'test_wikipedia_300_passlen_18_entrymin_250', 'test_twitter_200_passlen_18_entrymin_250', 
	'test_wikipedia_50_passlen_18_entrymin_250', 'test_wikipedia_100_passlen_18_entrymin_250', 
	'test_wikipedia_200_passlen_18_entrymin_250', 'test_twitter_50_passlen_18_entrymin_250', 
	'forum_data']

	path = os.getcwd()

	if glove_set == "twitter":
		glove_data_file = path + "/glove.twitter.27B/glove.twitter.27B." + str(glove_dim) + "d.txt"
		if glove_dim == 50:
			db_index = 7
		elif glove_dim ==100:
			db_index = 1
		elif glove_dim == 200:
			db_index = 3
	elif glove_set == "wikipedia":
		glove_data_file = path + "/glove.6B/glove.6B." + str(glove_dim) + "d.txt"
		if glove_dim ==50:
			db_index =4
		elif glove_dim == 100:
			db_index = 5
		elif glove_dim == 200:
			db_index =6



	words, word_dict, vocabulary_size = import_vocab(db_collections[db_index])
	embedding_matrix = populate_embedding_matrix(glove_data_file, glove_dim, words, vocabulary_size)
	raw_dataframe = get_dataframe(db_collections[db_index])

	test_dataframe = format_content(word_dict, raw_dataframe)
	test_dataframe = shuffle(test_dataframe)


	string_labels = test_dataframe['labels'].unique()
	label_dict = {}
	for i in range(len(string_labels)):
		label_dict[string_labels[i]] = i
	labels = test_dataframe['labels'].map(label_dict)
	categorical_labels = labels

	data = np.array(test_dataframe['word_tokens'].tolist())

	return data, vocabulary_size, categorical_labels, embedding_matrix


def eval_classifications(assignments, classifications):
	"""Compare the assignments and classifications to find errors"""
	correct = 0
	assignments_list = list(assignments)
	classifications_list = list(classifications)
	for i in range(len(assignments)):
		assign_list = list(assignments[i])
		class_list = list(classifications[i])

		maxpredict = assign_list.index(max(assign_list))
		classindx = class_list.index(max(class_list))
		if maxpredict == classindx:
			correct += 1

	return correct/len(assignments)


def run_cross_validation(epochs,k, glove_sets, glove_dims, optimizers, losses, activations, hidden_layers, dropouts):
	best_glove_set = ""
	best_glove_dim = ""
	best_optimizer = ""
	best_activation = ""
	best_hidden_layer = 0
	best_dropout = 0
	best_loss = ""
	best_option = 0
	best_result = 0
	accuracies = []
	option = 0
	for glove in glove_sets:
		for dim in glove_dims:
			data, vocabulary_size, categorical_labels, embedding_matrix = create_dataframe(glove, dim)
			validation_sets, valid_target, training_sets, train_target = generate_subsets(data, k, categorical_labels)
			for optimizer in optimizers:
				for loss in losses:
					for activation in activations:
						for hidden_layer in hidden_layers:
							for dropout in dropouts:
								accuracy_val = 0
								for i in range(k):
									glove_model = create_glove_model(vocabulary_size, dim, embedding_matrix, activation, loss, optimizer, hidden_layer, dropout)
									glove_model.fit(np.array(training_sets[i]), np.array(train_target[i]), epochs = epochs)
									predictions= glove_model.predict(np.array(validation_sets[i]))
									accuracy = eval_classifications(predictions, np.array(valid_target[i]))
									print ('accuracy', accuracy)
									print(glove_model.summary())
									if accuracy > best_result:
											best_glove_set = glove
											best_glove_dim = dim
											best_optimizer = optimizer
											best_activation = activation
											best_hidden_layer = hidden_layer
											best_dropout = dropout
											best_loss = loss
											best_result = accuracy
											best_option = option
									accuracy_val += accuracy
								accuracies.append(accuracy_val/k)

							option += 1

	print('best_glove_set:', best_glove_set)
	print('best_glove_dim:', best_glove_dim)
	print('best_hidden_layer:', best_hidden_layer)
	print('best_optimizer:', best_optimizer)
	print('best activation:', best_activation)
	print('best dropout:', best_dropout)
	print('best_loss:', best_loss)
	print('BEST ACCURACY:', best_result)

	with open("results.csv", 'w') as file:
		file.write("best glove set, best glove dimension, best hidden layer, best_optimizer, best_activation, best_dropout, best_result, best option, accuracies")
		file.write(str(best_glove_set) +"," + str(best_glove_dim) + "," + str(best_hidden_layer) + "," + str(best_optimizer) + "," + str(best_activation) +"," + str(best_dropout) +"," + str(best_loss) + "," + str(best_result) +"," + str(best_option) + "," + str(accuracies))#CHECK THIS

	return accuracies, best_option, best_result


glove_sets = ["twitter"]
glove_dims = [200]
losses = ["categorical_crossentropy"]
optimizers = ["adam"]
activations = ["softmax"]
dropouts = [0.1, 0.2]
hidden_layers = [50, 100]
epochs = 100
k=5


run_cross_validation(epochs,k, glove_sets, glove_dims, optimizers, losses, activations, hidden_layers, dropouts)

