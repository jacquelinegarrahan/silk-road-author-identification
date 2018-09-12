import pandas as pd
import csv
import os

def create_dataframe(glove_data_file):
	dataframe = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	values = list(dataframe.index)
	return dataframe, values

def check_label(w, values):
	if w in values:
		return True
	else:
		return False

def vec(word, dataframe):
	vector = dataframe.loc[word].as_matrix()
	return vector

def create_glove_entry(post, create_dataframe):
	vector = [] * len(post)
	for word in post:
		vector[i] = vec(word, dataframe)

if __name__ == "__main__":
	path = os.getcwd()
	glove_data_file = path + "/glove.twitter.27B/glove.twitter.27B.100d.txt"

	vec = vec("hello", glove_data_file)
	print(vec)