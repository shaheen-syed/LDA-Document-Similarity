# coding: utf-8

"""
	Created by Shaheen Syed
	Date: 06 June 2019
"""

import logging # logging to console and file
import os
import glob2 # read directories
import sys
import csv # handle csv files
import textract # for pdf to plain text conversion
import spacy # for NLP
import numpy as np # for lin. algebra
from nltk.corpus import stopwords # more stopwords
from datetime import datetime
from gensim import corpora, models # for latent dirichlet allocation

def set_logger(folder_name = 'logs'):

	"""
		Set up the logging to console layout

		Parameters
		----------
		folder_name : string, optional
				name of the folder where the logs can be saved to

	"""

	# create the logging folder if not exists
	create_directory(folder_name)

	# define the name of the log file
	log_file_name = os.path.join(folder_name, '{:%Y%m%d%H%M%S}.log'.format(datetime.now()))

	# set up the logger layout to console
	logging.basicConfig(filename=log_file_name, level=logging.NOTSET)
	console = logging.StreamHandler()
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	logger = logging.getLogger(__name__)


def create_directory(name):

	"""
		Create directory if not exists

		Parameters
		----------
		name : string
				name of the folder to be created

	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.info('Created directory: {}'.format(name))
	except Exception, e:
		logging.error('[createDirectory] : {}'.format(e))
		exit(1)


def read_directory(directory):

	"""

		Read file names from directory recursively

		Parameters
		----------
		directory : string
					directory/folder name where to read the file names from

		Returns
		---------
		files : list of strings
    			list of file names
	"""
	
	try:
		return glob2.glob(os.path.join( directory, '**' , '*.*'))
	except Exception, e:
		logging.error('[read_directory] : {}'.format(e))
		exit(1)


def save_csv(data, name, folder):

	"""
		Save list of list as CSV (comma separated values)

		Parameters
		----------
		data : list of list
    			A list of lists that contain data to be stored into a CSV file format
    	name : string
    			The name of the file you want to give it
    	folder: string
    			The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# create the path name (allows for .csv and no .csv extension to be handled correctly)
		suffix = '.csv'
		if name[-4:] != suffix:
			name += suffix

		# create the file name
		path = os.path.join(folder, name)

		# save data to folder with name
		with open(path, "w") as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(data)

	except Exception, e:
		logging.error('[save_csv] : {}'.format(e))
		exit(1)



def read_csv(filename, folder = None):

	"""
		Read CSV file and return as a list

		Parameters
		---------
		filename : string
			name of the csv file
		folder : string (optional)
			name of the folder where the csv file can be read

		Returns
		--------

	"""

	if folder is not None:
		filename = os.path.join(folder, filename)
	
	try:
		# increate CSV max size
		csv.field_size_limit(sys.maxsize)
		
		# open the filename
		with open(filename, 'rb') as f:
			# create the reader
			reader = csv.reader(f)
			# return csv as list
			return list(reader)
	except Exception, e:
		logging.error('Unable to open CSV {} : {}'.format(filename, str(e)))


def save_dic_to_csv(dic, file_name, folder):

	"""
		Save a dictionary as CSV (comma separated values)

		Parameters
		----------
		dic : dic
    			dictionary with key value pairs
    	name : string
    			The name of the file you want to give it
    	folder: string
    			The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# check if .csv is used as an extension, this is not required
		if file_name[-4:] == '.csv':
			file_name = file_name[:-4]

		# create the file name
		file_name = os.path.join(folder, file_name + '.csv')

		# save data to folder with name
		with open(file_name, "w") as f:

			writer = csv.writer(f, lineterminator='\n')
			
			for k, v in dic.items():
				writer.writerow([k, v])

	except Exception, e:
		logging.error('[save_dic_to_csv] : {}'.format(e))
		exit(1)

def pdf_to_plain(pdf_file):

	
	"""
		Read PDF file and convert to plain text

		Parameters
		----------
		pdf_file : string
			location of pdf file

		Returns
		---------
		plain_pdf = string
			plain text version of the PDF file.
	"""


	try:

		# use textract to convert PDF to plain text
		return textract.process(pdf_file, encoding='utf8')

	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		return None

def full_text_preprocessing(content):
	
	"""
		Some pre-processing of the PDF full-text

		Parameters
		----------
		content : string
			plain text content of a PDF file

		Returns
		---------
		content : string
			somewhat cleaned up version of the PDF content

	"""

	try:

		# make sure content is unicode
		content = content.decode('utf-8')

		# fix soft hyphen
		content = content.replace(u'\xad', "-")
		# fix em-dash
		content = content.replace(u'\u2014', "-")
		# fix en-dash
		content = content.replace(u'\u2013', "-")
		# minus sign
		content = content.replace(u'\u2212', "-")
		# fix hyphenation that occur just before a new line
		content = content.replace('-\n','')
		# remove new lines/carriage returns
		content = content.replace('\n',' ')

		# correct for ligatures
		content = content.replace(u'\u010f\u0179\x81', 'fi') # fi ligature
		content = content.replace(u'\u010f\u0179\x82', 'f') # weird f
		content = content.replace(u'\ufb02', "fl")	# fl ligature
		content = content.replace(u'\ufb01', "fi")	# fi ligature
		content = content.replace(u'\ufb00', "ff")	# ff ligature
		content = content.replace(u'\ufb03', "ffi") # ffi ligature
		content = content.replace(u'\ufb04', "ffl") # ffl ligature

		return content
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit()


def save_plain_text(plain_text, file_name, folder):

	"""
		Save string as text file

		Parameters
		----------
		plain_text : string
    		plain text to save
    	file_name : string
    			The name of the file you want to give it
    	folder: string
    			The folder location

	"""

	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# check if .txt is used as an extension, this is not required
		if file_name[-4:] == '.txt':
			file_name = file_name[:-4]

		# create the file name
		file_name = os.path.join(folder, file_name + '.txt')

		# save data to folder with name
		with open(file_name, "w") as f:

			# write plain text to file
			f.write(plain_text.encode('utf-8'))

	except Exception, e:
		logging.error('[save_plain_text] : {}'.format(e))
		exit(1)

def read_plain_text(file_name):

	"""
		Save string as text file

		Parameters
		----------
    	file_name : string
    			The name of the file you want to give it

    	Returns
    	--------
    	plain_text : string
    		the plain text from the .txt file

	"""

	try:

		# save data to folder with name
		with open(file_name, 'rb') as f:

			# read the content and return
			return f.read()

	except Exception, e:
		logging.error('[read_plain_text] : {}'.format(e))
		exit(1)

def setup_spacy():

	# setting up spacy and loading an English corpus
	nlp = spacy.load('en')

	# load the same corpus but in a different way (depends if there is a symbolic link)
	#nlp = spacy.load('en_core_web_sm')

	# add some more stopwords; apparently spacy does not contain all the stopwords
	for word in set(stopwords.words('english')):

		nlp.Defaults.stop_words.add(unicode(word))
		nlp.Defaults.stop_words.add(unicode(word.title()))

	for word in nlp.Defaults.stop_words:
		lex = nlp.vocab[word]
		lex.is_stop = True
	
	return nlp

def word_tokenizer(text):

	"""
		Function to return individual words from text. Note that lemma of word is returned excluding numbers, stopwords and single character words

		Parameters
		----------
		text : spacy object
			plain text wrapped into a spacy nlp object

		Returns
	"""

	# start tokenizing
	try:
		# Lemmatize tokens, remove punctuation, remove single character tokens and remove stopwords.
		return  [token.lemma_ for token in text if token.is_alpha and not token.is_stop and len(token) > 1]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def remove_domain_specific_stopwords(tokens):

	"""
		Remove domain specific stop words

		Parameters
		-----------
		tokens: list
			list of tokens

		Returns
		-----------
		tokens : list
			filtered list of tokens


	"""

	stopwords = ["doi", "faf", "ghoti", "fi", "ltd", "pp", "no", "blackwell", "fig", "de", "john", "wiley", "sons", "ices", "eds", "fao", "ii", "van", "la", "et",
					 "ed", "iii", "and/or", "sp", "appendix", "vs", "st", "ab", "du", "he", "des", "fl", "ne", "le", "el", "ec", "dc", "se", "ad", "nj", "th", "cid", "ha",
					 "tl", "sd", "la", "spp", "se", "e-mail", "st", "fi", "eg", "ie", "abstract", "introduction", "correspondence", "author", "figure", "fig", "table",
					 "published", "http", "www", "al", "per", "cowx", "acknowledgments", "nn", "nnn", "bulletin", "sci", "sh", "sl", "re", "von", "jr", "inc", "vol", "figs",
					 "df", "springer-verlag", "springer", "verlag", "ab", "abc", "january", "february", "march", "april", "may", "june", "july", "august", "september", 
					 "october", "november", "december", "bull", "introduction", "results", "discussion", "rev", "jj", "conclusion", "conclusions", "chapman", "hall",
					 "contents", "page", "section", "chapter", "summary", "appendix", "references", "discussion", "keywords", "accepted", "received", "crossmark", "suppl",
					 "fog", "elsevier", "ssdi", "pii", "crown", "copyright", "fme", "bv", "by", "fax", "tel", "sciencedirect", "volume", "nrc", "print", "online", "issn", "doi",
					 "mar", "corresponding", "article", "address", "among", "amongst", "within", "using", "used", "use", "with"]

	# use only the tokens that are not in the list of stopwords
	tokens = [t for t in tokens if t not in stopwords]

	return tokens



def get_dic_corpus(file_folder):

	"""
		Read dictionary and corpus for Gensim LDA

		Parameters
		-----------
		file_folder : os.path
			locatino of dictionary and corpus

		Returns
		dictionary : dict()
			LDA dictionary
		corpus : mm
			LDA corpus
	"""

	# create full path of dictionary
	dic_path = os.path.join(file_folder, 'dictionary.dict')
	# create full path of corpus
	corpus_path = os.path.join(file_folder, 'corpus.mm')


	# check if dictionary exists
	if os.path.exists(dic_path):
		dictionary = corpora.Dictionary.load(dic_path)
	else:
		logging.error('LDA dictionary not found')
		exit(1)

	# check if corpus exists
	if os.path.exists(corpus_path):
		corpus = corpora.MmCorpus(corpus_path)
	else:
		logging.error('LDA corpus not found')
		exit(1)

	return dictionary, corpus


def load_lda_model(model_location):

	"""
		Load the LDA model

		Parameters
		-----------
		model_location : os.path()
			location of LDA Model

		Returns
		-------
		model : gensim.models.LdaModel
			trained gensim lda model
	"""

	model_path = os.path.join(model_location, 'lda.model')

	if os.path.exists(model_path):
		return  models.LdaModel.load(model_path)
	else:
		logging.error('LDA model not found')
		exit(1)

def getTopicLabel(topic, lda_type):

	"""
		Return topic label

		Parameters
		-----------
		topic : int
			topic id from lda model
		lda_type: int
			type of lda model, either 1 or 2

		Returns
		-------
		label: string
			label for topic word distribution

	"""

	topic_labels_1 = {
						0: 'Diseases', 
						1: 'Reproduction', 
						2: 'Habitats', 
						3: 'Salmonids', 
						4: 'Genetics', 
						5: 'Climate effects', 
						6: 'Models (estimation & stock)', 
						7: 'Age & growth', 
						8: 'Diet', 
						9: 'Aquaculture (growth effects)', 
						10: 'Physiology', 
						11: 'Immunogenetics', 
						12: 'Aquaculture (health effects)', 
						13: 'Shellfish', 
						14: 'Gear technology & bycatch', 
						15: 'Management'
					}

	topic_labels_2 = {		
						0 : 'Conservation',
						1 : 'Morphology',
						2 : 'Salmon',
						3 : 'Reproduction',
						4 : 'Non-Fish Species',
						5 : 'Corals',
						6 :  'Biochemistry',
						7 : 'Freshwater',
						8 : 'Diet',
						9 : 'North Atlantic',
						10 : 'Southern Hemisphere',
						11 : 'Development',
						12 : 'Genetics',
						13 : 'Assemblages',
						14 : 'Growth Experiments',
						15 : 'Stock Assessment',
						16 : 'Growth',
						17 : 'Tracking and Movement',
						18 : 'Fishing Gear',
						19 : 'Primary Production',
						20 : 'Models',
						21 : 'Salmonids',
						22 : 'Acoustics and Swimming',
						23 : 'Estuaries',
						24 : 'Fisheries Management'
					}


	if lda_type == 1:

		return topic_labels_1[topic]

	elif lda_type == 2:

		return topic_labels_2[topic]
	else:
		logging.error('Unknown lda_type given: {}'.format(lda_type))
		exit(1)

def calculate_hellinger_distance(p, q):

	"""
		Calculate the hellinger distance between two probability distributions

		note that the hellinger distance is symmetrical, so distance p and q = q and p

		other measures, such as KL-divergence, are not symmetric but can be used instead

		Parameters
		-----------
		p : list or array
			first probability distribution
		q : list or array
			second probability distribution

		Returns
		--------
		hellinger_dinstance : float
			hellinger distance of p and q

	"""

	return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

