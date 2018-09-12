from html.parser import HTMLParser
from bs4 import BeautifulSoup
import os


def cleanHTML(file, standard_filename, index):
	"""Deletes super annoying Simple Machines footer"""
	file_data = open(file, "r")
	soup = BeautifulSoup(file_data, 'html.parser')
	if soup.find('div', id="footer") != None:
		soup.find('div', id="footer").decompose()
	#delete file and replace with new version
	os.remove(file)
	with open(standard_filename +  '_' + str(index) + '.html', "w") as outfile:
		outfile.write(soup.prettify())
	return

def cleanDirectory(directory_path, standard_filename):
	#iterate over filelist
	path = os.getcwd() + directory_path
	file_list = os.listdir(path)
	for f in range(len(file_list)):
		cleanHTML(path +'/' + file_list[f], path + '/' + standard_filename , f )
	return 

if __name__ == "__main__":
	cleanDirectory("/test_files", "test_file")