import pymongo

def reset_db():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	myclient.drop_database('dnm_forums')
	print("Database deleted.")
	return

if __name__ == "__main__":
	reset_db()