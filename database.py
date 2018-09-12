import pymongo
from parse_file import parse_forum_html 
from glove import vec, check_label, create_dataframe
import os 
from numpy import mean, median
import pandas as pd


def create_db():
	"""Checks if database exists and creates new one if not"""
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	dblist = myclient.list_database_names()
	db = myclient["dnm_forums"]
	initial_collection = db["forum_data"]
	return db, myclient, initial_collection

def access_existing_db():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	dblist = myclient.list_database_names()
	db = myclient["dnm_forums"]
	initial_collection = db["forum_data"]
	return db, myclient, initial_collection

def create_test_collection(db, entry_filter, passage_length, glove_set, glove_dim):
	"""Creates a test collection with vectors of specific length"""
	if glove_set == "twitter":
		glove_data_file = path + "/glove.twitter.27B/glove.twitter.27B." + str(glove_dim) + "d.txt"
	elif glove_set == "wikipedia":
		glove_data_file = path + "/glove.6B/glove.6B." + str(glove_dim) + "d.txt"

	collections = db.list_collection_names()
	test_collection_name = "test_" + glove_set + '_' +str(glove_dim)+ "_passlen_" +  str(passage_length) + "_entrymin_" + str(entry_filter)
	if test_collection_name in collections:
		db.collection.drop(test_collection_name)

	test_collection = db[test_collection_name]
	return test_collection, glove_data_file


def pop_initial_collection(db_collection, directory_path):
	file_list = os.listdir(directory_path)
	for f in range(len(file_list)):
		print('adding %s to database' %f)
		authors, content = parse_forum_html(directory_path +'/' + file_list[f])
		for i in range(len(authors)):
		#Check if author already exists and append
			if alreadyExists(db_collection, authors[i]):
				db_collection.update({"author":authors[i]}, {"$addToSet": {"content": content[i]}})
			else:
				db_collection.insert({"author": authors[i], "content": [content[i]]})
	print('Populating database complete!')


def update_content(db_collection, author, new_post):
	"""If author already exists, update with new content"""
	db_collection.update({'author': author}, {'$push': {'content': new_post}})


def check_content(db_collection, author):
	"""Gets content from one author and checks it"""
	print(db_collection.find_one({'author': author}))
	content = db_collection.content
	print(content)


def convert_passage_to_list(content):
	list = content.split()
	return list


def alreadyExists(db_collection, new_author):
	entry =  db_collection.find_one({"author": { "$in": [new_author]}}) != None
	return entry



def pop_test_collection(test_collection, initial_collection, passage_length_filter, entry_filter, glove_data_file):
	words, values = create_dataframe(glove_data_file)
	authors = initial_collection.distinct('author')

	filtered_authors = []

	for author in authors:
		print(author)
		entry = initial_collection.find_one({"author": { "$in": [author]}})
		full_entry = []
		filtered_entry = []

		for entry in entry["content"]:
			full_entry += convert_passage_to_list(entry)
		print(full_entry)

		n = 0
		if len(full_entry) >= entry_filter * passage_length_filter:
			for word in full_entry:
				if check_label(word, values):
					filtered_entry += [word]

					##NEED TO CONVERT THE WORDS INTO INDICES AND CREATE VOCABULARY

		new_entries = []

		if len(filtered_entry) >= entry_filter * passage_length_filter:
			while len(filtered_entry) > (n+1) * passage_length_filter:
				new_entries.append(filtered_entry[n * passage_length_filter : (n+1) * passage_length_filter]) 
				n += 1
			test_collection.insert({"author": author, "content": new_entries})

	authors_check = test_collection.distinct('author')
	print('number authors', len(authors_check))


def check_vals(entry_filter, passage_length_filter, glove_data_file):
	words, values = create_dataframe(glove_data_file)
	db = myclient["dnm_forums"]
	print(db.list_collection_names())
	initial_collection = db["forum_data"]
	authors = initial_collection.distinct('author')
	num_entries = []
	for author in authors:
		entries = initial_collection.find_one({"author": { "$in": [author]}})
		full_entry = []
		filtered_entry = []

		for entry in entries["content"]:
			full_entry += convert_passage_to_list(entry)
		
		if len(full_entry)/passage_length_filter >= entry_filter:
			for word in full_entry:
				if check_label(word, values):
					filtered_entry += word

		num_entries.append(int(len(filtered_entry)/passage_length_filter))


	authors_check = []
	for i in range(len(authors)):
		if num_entries[i] >= entry_filter:
			authors_check.append(authors[i])

	print('max entry:', max(num_entries))

	print('distinct authors', len(authors_check))



def get_vocab(collection_name):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	db = myclient["dnm_forums"]
	collection = db[collection_name]

	#Drop prior vocab
	vocab_collection = db["vocab_collection"]
	db.vocab_collection.drop()
	vocab_collection = db["vocab_collection"]

	authors = collection.distinct("author")
	for author in authors:
		entry = collection.find_one({"author": { "$in": [author]}})
		for item in entry["content"]:
			#adjusted for kind of sh*t setup with nested loop
			new_item = item[0]
			for word in new_item:
				vocab_collection.insert({"word": word})

	words = vocab_collection.distinct("word")
	return words, len(words)


def get_dataframe(collection_name):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	db = myclient["dnm_forums"]
	collection = db[collection_name]
	data = pd.DataFrame(list(collection.find()))
	return data
		
		


if __name__ == "__main__":
	twitter_glove_dims = [25, 50, 100, 200]
	wiki_glove_dims = [50, 100, 200, 300]
	passage_lengths = [18, 50]

	db, myclient, initial_collection = access_existing_db()
	path = os.getcwd() 
	entry_filter = 250
	passage_length_filter = 18
#	glove_set = "twitter"
#	glove_dim = 50
	dirpath = path + '/test_files'
	pop_initial_collection(initial_collection, dirpath)
#	if glove_set == "twitter":
#		glove_data_file = path + "/glove.twitter.27B/glove.twitter.27B." + str(glove_dim) + "d.txt"
#	elif glove_set == "wikipedia":
#		glove_data_file = path + "/glove.6B/glove.6B." + str(glove_dim) + "d.txt"


	#check_vals(entry_filter, passage_length_filter, glove_data_file)

	for dim in twitter_glove_dims:
		glove_set = "twitter"
		test_collection, glove_data_file = create_test_collection(db, entry_filter, passage_length_filter, glove_set, dim)
		pop_test_collection(test_collection, initial_collection, passage_length_filter, entry_filter, glove_data_file)
#
	for dim in wiki_glove_dims:
		glove_set = "wikipedia"
		test_collection, glove_data_file = create_test_collection(db, entry_filter, passage_length_filter, glove_set, dim)
		pop_test_collection(test_collection, initial_collection, passage_length_filter, entry_filter, glove_data_file)
#



#	test_collection, glove_data_file = create_test_collection(db, entry_filter, passage_length_filter, glove_set, glove_dim)
#	pop_test_collection(test_collection, initial_collection, passage_length_filter, entry_filter, glove_data_file)







